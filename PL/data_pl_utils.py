import numpy as np
import pandas as pd
import os
import torchvision

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split


def make_expanded_data_loader(
        source_model, tr_loader, unlab_loader,
        threshold_quantile=0.95,
        trch_prng=None,
        verbose=False,
        ):
    # Extract full unlabeled data as tensor
    xu_N2 = unlab_loader.dataset.dataset.x_N2
    # Build pseudo-labeled subset
    xnew_X2, ynew_X = make_pseudolabels_for_most_confident_fraction(
        source_model, xu_N2,
        threshold_quantile=threshold_quantile,
        trch_prng=trch_prng,
        verbose=verbose)

    # Concatenate this new subset with existing train set
    xtr_N2 = tr_loader.dataset.dataset.x_N2[tr_loader.dataset.indices]
    ytr_N = tr_loader.dataset.dataset.targets[tr_loader.dataset.indices]
    expanded_dset = torch.utils.data.TensorDataset(
        torch.cat([xtr_N2, xnew_X2]),
        torch.cat([ytr_N, ynew_X]),
        )
    # Create and return DataLoader
    expanded_loader = torch.utils.data.DataLoader(
        expanded_dset,
        batch_size=tr_loader.batch_size,
        shuffle=True)
    return expanded_loader

def make_pseudolabels_for_most_confident_fraction(
        source_model, xu_N2,
        threshold_quantile=0.5,
        trch_prng=None,
        verbose=False):
    ''' Create pseudolabeled version of provided dataset.

    Obtains pseudolabels and associated confidences for each instance.
    Then, for each label, identifies label-specific threshold for desired
    quantile. Keeps only those instances with confidence above that threshold.

    Returns
    -------
    xnew_XF : torch tensor, shape (X, F)
        Equal to subset of provided xu tensor
    ynew_X : torch tensor, shape (X,)
        Corresponding pseudolabels for each row of xnew_XF
    '''
    N = xu_N2.shape[0]
    with torch.no_grad():
        probs = source_model(xu_N2)
        phat_N, yhat_N = torch.max(probs, dim = 1)

    phat_N = phat_N.detach()
    yhat_N = yhat_N.detach()

    # Report on how unique different predicted probas are across dataset
    if verbose:
        uvals, counts = np.unique(
            phat_N.numpy().round(3),
            return_counts=True)
        U = len(uvals)
        print("Unlabeled set of size %3d maps to %3d unique 3-digit probas" % (
            N, U))
        for ii, (u, c) in enumerate(zip(uvals, counts)):
            if ii < 4:
                print("%3d %s" % (c,u))
            elif ii == 4:
                print("%3d %s" % (c,u))
                print("...")
            elif ii >= U - 4:
                print("%3d %s" % (c,u))
    
    # Find class-specific thresholds
    pthresh0 = torch.quantile(phat_N[yhat_N==0], threshold_quantile)
    pthresh1 = torch.quantile(phat_N[yhat_N==1], threshold_quantile)

    # keepmask0_N is bool indicator of whether both are true
    # * yhat == 0 for example i
    # * predicted probability is above the threshold
    keepmask0_N = torch.logical_and(phat_N >= pthresh0, yhat_N == 0)
    keepmask1_N = torch.logical_and(phat_N >= pthresh1, yhat_N == 1)
    size0 = int((1 - threshold_quantile) * N / 2)
    size1 = int(size0)

    # Break ties, in case many examples are "tied"
    # We only want a subset of the target size, not more than expected.
    if torch.sum(keepmask0_N) > size0:
        ids0 = torch.nonzero(keepmask0_N, as_tuple=True)[0]
        perm = torch.randperm(ids0.size(0), generator=trch_prng)
        ids0 = ids0[perm[:size0]]
        keepmask0_N[:] = 0
        keepmask0_N[ids0] = 1
    if torch.sum(keepmask1_N) > size1:
        ids1 = torch.nonzero(keepmask1_N, as_tuple=True)[0]
        perm = torch.randperm(ids1.size(0), generator=trch_prng)
        ids1 = ids1[perm[:size1]]
        keepmask1_N[:] = 0
        keepmask1_N[ids1] = 1

    # Create new tensors to represent x and y for pseudolabeled subset
    xnew_X2 = xu_N2[torch.logical_or(keepmask0_N, keepmask1_N)]
    ynew_X = yhat_N[torch.logical_or(keepmask0_N, keepmask1_N)]
    return xnew_X2, ynew_X


# Constants for image preprocessing
mean_pn_RGB_3 = [0.485, 0.456, 0.406]
stddev_pn_RGB_3 = [0.229, 0.224, 0.225]

DEFAULT_IM_PREPROCESSING = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean_pn_RGB_3, std=stddev_pn_RGB_3),
])

IM_PREPROCESSING_FOR_VIEW = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
])

AUGMENTATION_TRANSFORMS = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    torchvision.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean_pn_RGB_3, std=stddev_pn_RGB_3),
])

class PNDataset(torchvision.datasets.ImageFolder):
    def __init__(
        self, 
        root, 
        transform=None, 
        target_transform=None, 
        n_samples_per_class=None
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.n_samples_per_class = n_samples_per_class
        
        if self.n_samples_per_class is not None:
            self._filter_samples()
    
    def transform_for_viz(self, x):
        return IM_PREPROCESSING_FOR_VIEW(x)
    
    def _filter_samples(self):
        class_counts = {}
        filtered_samples = []
        
        for sample, target in self.samples:
            if target not in class_counts:
                class_counts[target] = 0
                
            if class_counts[target] < self.n_samples_per_class:
                filtered_samples.append((sample, target))
                class_counts[target] += 1
                
        self.samples = filtered_samples
        self.targets = [target for _, target in filtered_samples]

def create_mixmatch_loaders(
    train_loader: DataLoader,
    unlabeled_frac: float = 0.8,
    random_state: int = 1234
):
    """
    Create labeled and unlabeled data loaders for MixMatch from a training loader
    """
    # Collect all data from the loader
    all_data = []
    all_targets = []
    
    for data, targets in train_loader:
        all_data.append(data)
        all_targets.append(targets)
    
    all_data = torch.cat(all_data)
    all_targets = torch.cat(all_targets)
    
    # Calculate split sizes
    total_size = len(all_data)
    unlabeled_size = int(total_size * unlabeled_frac)
    labeled_size = total_size - unlabeled_size
    
    # Create train/unlabeled split
    indices = torch.randperm(total_size)
    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:]
    
    # Create datasets
    labeled_data = all_data[labeled_indices]
    labeled_targets = all_targets[labeled_indices]
    unlabeled_data = all_data[unlabeled_indices]
    unlabeled_targets = torch.zeros_like(all_targets[unlabeled_indices])
    
    labeled_dataset = TensorDataset(labeled_data, labeled_targets)
    unlabeled_dataset = TensorDataset(unlabeled_data, unlabeled_targets)
    
    # Create loaders with same batch size as original
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )
    
    return labeled_loader, unlabeled_loader