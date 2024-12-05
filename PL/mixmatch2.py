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
        
    def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        ws = AverageMeter()
        end = time.time()

        bar = Bar('Training', max=args.train_iteration)
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)

        model.train()
        for batch_idx in range(args.train_iteration):
            try:
                inputs_x, targets_x = labeled_train_iter.next()
            except Exception as _:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()

            try:
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
            except Exception as _:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

            # measure data loading time
            data_time.update(time.time() - end)

            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)

            if use_cuda:
                inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
                inputs_u = inputs_u.cuda()
                inputs_u2 = inputs_u2.cuda()


            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p**(1/args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l = np.random.beta(args.alpha, args.alpha)

            l = max(l, 1-l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.train_iteration)

            loss = Lx + w * Lu

            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Lx.item(), inputs_x.size(0))
            losses_u.update(Lu.item(), inputs_x.size(0))
            ws.update(w, inputs_x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                        batch=batch_idx + 1,
                        size=args.train_iteration,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        w=ws.avg,
                        )
            bar.next()
        bar.finish()

        return (losses.avg, losses_x.avg, losses_u.avg,)

    def validate(valloader, model, criterion, epoch, use_cuda, mode):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        bar = Bar(f'{mode}', max=len(valloader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # measure data loading time
                data_time.update(time.time() - end)

                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                # compute output
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx + 1,
                            size=len(valloader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            )
                bar.next()
            bar.finish()
        return (losses.avg, top1.avg)

    def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

    def linear_rampup(current, rampup_length=args.epochs):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    class SemiLoss(object):
        def __call__(self, outputs_x, y, outputs_u, targets_u, epoch):
            probs_u = torch.softmax(outputs_u, dim=1)

            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * y, dim=1))
            Lu = torch.mean((probs_u - targets_u)**2)

            return Lx, Lu, args.lambda_u * linear_rampup(epoch)

    class WeightEMA(object):
        def __init__(self, model, ema_model, alpha=0.999):
            self.model = model
            self.ema_model = ema_model
            self.alpha = alpha
            self.params = list(model.state_dict().values())
            self.ema_params = list(ema_model.state_dict().values())
            self.wd = 0.02 * args.lr

            for param, ema_param in zip(self.params, self.ema_params):
                param.data.copy_(ema_param.data)

        def step(self):
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.params, self.ema_params):
                if ema_param.dtype==torch.float32:
                    ema_param.mul_(self.alpha)
                    ema_param.add_(param * one_minus_alpha)
                    # customized weight decay
                    param.mul_(1 - self.wd)

    def interleave_offsets(batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets


    def interleave(xy, batch):
        nu = len(xy) - 1
        offsets = interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

