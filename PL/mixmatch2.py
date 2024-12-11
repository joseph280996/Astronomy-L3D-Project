import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from torchvision import transforms
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MixMatch:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        # ema_optimizer,
        num_augmentations: int = 2,
        temperature: float = 0.5,
        alpha: float = 0.75,
        lambda_u: float = 75,
        device: str = "mps",
        max_grad_norm: float = 1.0,
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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optimizer
        # self.ema_optimizer = ema_optimizer

        # Setup default augmentations
        self.augmentation_pool = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.ToTensor(),
            ]
        )
        self.xent_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l2_loss_fn = nn.MSELoss(reduction="sum")

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def train(self, labeled_trainloader, unlabeled_trainloader):

        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        ws = AverageMeter()

        self.model.train()
        for (inputs_x, targets_x), inputs_u in zip(labeled_trainloader, unlabeled_trainloader):

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            targets_x = torch.nn.functional.one_hot(targets_x, 2).float()

            inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(
                self.device, non_blocking=True
            )
            inputs_u = inputs_u[0].to(self.device)

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u = self.model(inputs_u)
                p = torch.softmax(outputs_u, dim=1) 
                pt = torch.pow(p, (1 / self.T))
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([targets_x, targets_u], dim=0)

            l = np.random.beta(self.alpha, self.alpha)

            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.shape[0])

            w_inputs = all_inputs[idx]
            w_targets = all_targets[idx]

            mixed_input = l * all_inputs + (1 - l) * w_inputs
            mixed_target = l * all_targets + (1 - l) * w_targets

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = self.interleave(mixed_input, batch_size)

            logits = [self.model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(self.model(input))

            # put interleaved samples back
            logits = self.interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = self.semi_loss(
                logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:]
            )

            loss = Lx + w * Lu

            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Lx.item(), inputs_x.size(0))
            losses_u.update(Lu.item(), inputs_x.size(0))
            ws.update(w, inputs_x.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.ema_optimizer.step()

        return (
            losses.avg,
            losses_x.avg,
            losses_u.avg,
        )

    def validate(self, valloader):

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # measure data loading time
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                # compute output
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, prec5 = (outputs == 
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        return (losses.avg, top1.avg)

    def semi_loss(self, outputs_x, y, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * y, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.lambda_u

    def interleave_offsets(self, batch, nu):
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
        xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]
