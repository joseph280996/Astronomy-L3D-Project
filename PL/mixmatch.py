import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import warnings
import logging
import data_pl_utils
from torch.optim.lr_scheduler import OneCycleLR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixMatch:
    def __init__(
        self,
        model: nn.Module,
        num_augmentations: int = 2,
        temperature: float = 0.5,
        alpha: float = 0.75,
        lambda_u: float = 100,
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
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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

          
        # Ensure x1 and x2 have the same shape
        assert x1.shape == x2.shape, f"Shape mismatch: x1 {x1.shape} vs x2 {x2.shape}"
        # Ensure y1 and y2 have the same shape
        assert y1.shape == y2.shape, f"Shape mismatch: y1 {y1.shape} vs y2 {y2.shape}"

        # Mix images
        x_mix = lam * x1 + (1 - lam) * x2
        
        # Mix labels separately
        y_mix = lam * y1 + (1 - lam) * y2
        
        return x_mix, y_mix

    def train(
        self,
        tr_loader: DataLoader,
        va_loader: DataLoader,
        te_loader: DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        l2pen_mag = 0.0,
        save_path: str = 'best_mixmatch_model.pth',
        optimizer: Optional[torch.optim.Optimizer] = None,
        do_early_stopping=True,
        n_epochs_without_va_improve_before_early_stop=15,
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
            if hasattr(self.model, 'trainable_params'):
                params = self.model.trainable_params.values()
            else:
                params = self.model.parameters()
            optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        tr_info = {'loss': [], 'acc': []}
        va_info = {'loss': [], 'acc': []}
        epochs = []

        best_va_loss = float('inf')

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
                total_sample = 0
                pbar_info['batch_done'] = 0
                
                
                # Train for one epoch
                for batch_idx, ((x, y), (u, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
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

                    x_mix, y_mix = self.mixup(x_b, y_oh, wx[idx[:x_b.size(0)]], wy[idx[:x_b.size(0)]])
                    u_mix, q_mix = self.mixup(u, q, wx[idx[x_b.size(0):]], wy[idx[x_b.size(0):]])

                    y_mix_pred = self.model(x_mix)
                    q_mix_pred_logits = self.model(u_mix)

                    # Use the mask to compute supervised loss only on relevant examples
                    Lx = self.xent_fn(
                        y_mix_pred, 
                        y_mix,
                    )

                    # Unsupervised loss on the rest
                    Lu = self.l2_loss_fn(
                        F.softmax(q_mix_pred_logits),
                        q_mix
                    )
                    
                    loss = Lx + self.lambda_u * Lu
                    
                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar_info['batch_done'] += 1
                    progressbar.set_postfix(pbar_info)
                    
                    # Record losses
                    # Track training metrics
                    tr_loss += loss.item()
                    total_sample += x_b.size(0)

                # Training metrics
                tr_loss /= total_sample
                tr_loss /= pbar_info['batch_done']
            else:
                tr_loss = np.nan
            
            # Validation phase
            self.model.eval()
            total_loss = 0
            correct = 0
            total_sample = 0
            
            with torch.no_grad():
                for inputs, targets in va_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    batch_size = inputs.size(0)
                    
                    outputs = self.model(inputs)
                    
                    # Option 1: Using same loss as training
                    # targets_one_hot = F.one_hot(targets, num_classes=2).float()
                    # loss = F.cross_entropy(outputs, targets_one_hot)
                    
                    # Option 2: Using standard cross entropy (current approach)
                    loss = F.cross_entropy(outputs, targets, reduction='mean')
                    
                    total_loss += loss.item() * batch_size  # Weight by batch size
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total_sample += inputs.size(0)
            
            va_loss = total_loss / total_sample
            va_acc = 100. * correct / total_sample

            
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
    
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model on a dataset"""