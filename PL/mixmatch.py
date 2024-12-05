import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
import data_pl_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixMatch:
    def __init__(
        self,
        model: nn.Module,
        num_augmentations: int = 2,
        temperature: float = 0.5,
        alpha: float = 0.75,
        lambda_u: float = 75,
        device: str = 'cuda',
        max_grad_norm: float = 1.0
    ):
        """
        Implementation of MixMatch for semi-supervised learning
        
        Args:
            model: Neural network model
            num_augmentations: Number of augmentations for unlabeled data
            temperature: Sharpening temperature
            alpha: Beta distribution parameter for MixUp
            lambda_u: Unlabeled loss weight
            device: Device to use for computation
            max_grad_norm: Maximum norm for gradient clipping
        """
        self.model = model
        self.K = num_augmentations
        self.T = temperature
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup default augmentations
        self.augmentation_pool = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor()
        ])
        self.xent_fn = nn.CrossEntropyLoss(reduction = 'sum')
        self.l2_loss_fn = nn.MSELoss(reduction = 'sum')
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation with proper device handling"""
        augmented = self.augmentation_pool(x.cpu())
        return augmented.to(self.device)
    
    def sharpen(self, p: torch.Tensor) -> torch.Tensor:
        """Sharpen probability distributions"""
        p_temp = p ** (1 / self.T)
        return p_temp / p_temp.sum(dim=1, keepdim=True)
    
    def guess_labels(self, ub: torch.Tensor) -> torch.Tensor:
        """Generate pseudo-labels for unlabeled data"""
        with torch.no_grad():
            for _ in range(self.K):
                ub = self.augment(ub)
            pseudo_logits = F.softmax(self.model(ub), dim=1)
            pseudo_logits /= self.K
            return self.sharpen(pseudo_logits)
    
    def mixup(self, x1: torch.Tensor,  
            y1: torch.Tensor, 
            x2: torch.Tensor,
            y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform MixUp augmentation
        """
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        lam = max(lam, 1 - lam)
          
        # Mix images
        x_mix = lam * x1 + (1 - lam) * x2
        
        # Mix labels separately
        y_mix = lam * y1 + (1 - lam) * y2
        
        return x_mix, y_mix

    def train(
        self,
        tr_loader: DataLoader,
        va_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.0001,
        l2pen_mag = 0.1,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict:
        """
        Train the model using MixMatch with progress bars
        """

        labeled_loader, unlabeled_loader = data_pl_utils.create_mixmatch_loaders(
            train_loader=tr_loader,
            unlabeled_frac=0.8
        )

        # Setup optimizer and scheduler
        if optimizer is None:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate)
        
        tr_info = {'loss': [], 'acc': []}
        va_info = {'loss': [], 'acc': []}
        epochs = []

        best_va_loss = float('inf')

        n_train = float(len(tr_loader.dataset))
        n_valid = float(len(va_loader.dataset))
        
        # Create epoch progress bar
        progressbar = tqdm(range(num_epochs + 1))
        pbar_info = {}
        
        for epoch in progressbar:
            if epoch > 0:
                # Training phase
                self.model.train()
                tr_loss = 0.0
                tr_acc = 0.0
                pbar_info['batch_done'] = 0
                
                
                # Train for one epoch
                for _, ((x, y), (u, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
                    optimizer.zero_grad()

                    x_b = self.augment(x.to(self.device))
                    y = y.to(self.device)
                    u = u.to(self.device)
                    
                    # Generate pseudo-labels and convert targets to one-hot
                    q = self.guess_labels(u)
                    
                    # Concatenate and mix data
                    # In training loop:
                    wx = torch.cat([x_b, u], dim=0)
                    y_oh = torch.nn.functional.one_hot(y, 2).float()
                    wy = torch.cat([
                        y_oh,
                        q
                    ], dim=0)
                    
                    idx = torch.randperm(wx.shape[0])

                    x_mix, y_mix = self.mixup(wx, wy, wx[idx], wy[idx])

                    y_mix_pred = self.model(x_mix.to(self.device))

                    # Use the mask to compute supervised loss only on relevant examples
                    Lx = self.xent_fn(
                        y_mix_pred[:len(x_b)], 
                        y_mix[:len(x_b)],
                    )

                    # Unsupervised loss on the rest
                    Lu = self.l2_loss_fn(
                        F.softmax(y_mix_pred[len(x_b):]),
                        y_mix[len(x_b):]
                    )
                    
                    loss = Lx + self.lambda_u * Lu
                    
                    # Optimization step
                    loss.backward()
                    optimizer.step()

                    pbar_info['batch_done'] += 1
                    progressbar.set_postfix(pbar_info)
                    
                    # Record losses
                    # Track training metrics
                    tr_loss += loss.item()
                    tr_acc += (y_mix_pred == y_mix).sum().item()

                # Training metrics
                tr_loss /= n_train
                tr_acc /= n_train
            else:
                tr_loss = np.nan
                tr_acc = np.nan
            
            # Validation phase
            with torch.no_grad():
                self.model.eval()
                total_loss = 0
                correct = 0
                
                for inputs, targets in va_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    
                    # Option 1: Using same loss as training
                    # targets_one_hot = F.one_hot(targets, num_classes=2).float()
                    # loss = F.cross_entropy(outputs, targets_one_hot)
                    
                    # Option 2: Using standard cross entropy (current approach)
                    loss = F.cross_entropy(outputs, targets, reduction='sum')
                    
                    total_loss += loss.item()  # Weight by batch size
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
            
            va_loss = total_loss / n_valid
            va_acc = 100. * correct / n_valid

            epochs.append(epoch)
            tr_info['loss'].append(tr_loss)
            va_info['loss'].append(va_loss)
            va_info['acc'].append(va_acc)
            pbar_info.update({
                "tr_loss": tr_loss,
                "va_loss": va_loss, "va_acc": va_acc
            })
            
            # Update epoch progress bar
            progressbar.set_postfix(pbar_info)
            # Early stopping logic
            # If loss is dropping, track latest weights as best
            if va_loss < best_va_loss:
                best_epoch = epoch
                best_va_loss = va_loss
                best_va_acc = va_acc
                torch.save(self.model.state_dict(), "best_model.pth")
        
        # Close progress bars
        progressbar.close()
        print(f"Finished after epoch {epoch}, best epoch={best_epoch}")
        print("best va_xent %.3f" % best_va_loss)
        #print("best tr_err %.3f" % best_tr_err_rate)
        #print("best va_err %.3f" % best_va_err_rate)

        self.model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        result = {
            'lr':learning_rate, 'n_epochs':num_epochs, 'l2pen_mag':l2pen_mag,
            'tr':tr_info,
            'va':va_info,
            'best_va_acc': best_va_acc,
            'best_va_loss': best_va_loss,
            'best_epoch': best_epoch,
            'epochs': epochs
            }
        return self.model, result
    
    def interleave_offsets(self,batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets


    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    def linear_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current/rampup_length, 0.0, 1.0)
            return float(current)
