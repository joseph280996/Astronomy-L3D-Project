import numpy as np
import pandas as pd
import os

import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split

from sklearn.model_selection import train_test_split

mean_pn_RGB_3 = [0.485, 0.456, 0.406]
stddev_pn_RGB_3 = [0.229, 0.224, 0.225]

DEFAULT_IM_PREPROCESSING = torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)

labeled_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.Pad(padding=28),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
    ]
)

unlabeled_transform = torchvision.transforms.Compose(
    [    
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.Pad(padding=28),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
    ]
)

IM_PREPROCESSING_FOR_VIEW = torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)


class PNDataset(torchvision.datasets.ImageFolder):
    def __init__(
        self, root, transform=None, target_transform=None, n_samples_per_class=None
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

class TransformSubset:
    def __init__(self, subset, additional_transform):
        self.subset = subset
        self.additional_transform = additional_transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.additional_transform:
            x = self.additional_transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def make_PN_data_loaders_with_unlabeled(
        root=os.path.abspath('.'),
        transform=DEFAULT_IM_PREPROCESSING,
        target_transform=None,
        batch_size=32,
        n_samples_per_class_trainandvalid=250,
        frac_valid=0.2,
        frac_unlabeled=0.5,  # Fraction of training set to use as unlabeled
        random_state=23,
        verbose=True):
    PN_dev = PNDataset(os.path.join(root, 'train'), transform=transform)
    
    PN_test = PNDataset(os.path.join(root, 'test'), transform=transform)
        
    # Stratified sampling for train and val
    tr_idx, val_idx = train_test_split(np.arange(len(PN_dev)),
                                       test_size=frac_valid,
                                       random_state=random_state,
                                       shuffle=True,
                                       stratify=PN_dev.targets)

    # Create labeled and unlabeled subsets
    labeled_size = int(len(tr_idx) * (1 - frac_unlabeled))
    unlabeled_size = len(tr_idx) - labeled_size

    labeled_idx, unlabeled_idx = random_split(tr_idx, [labeled_size, unlabeled_size],
                                              generator=torch.Generator().manual_seed(random_state))

    # Subset for labeled data
    tr_labeled_set = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=labeled_transform), labeled_idx
    )

    # Subset for unlabeled data
    tr_unlabeled_set = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=unlabeled_transform), unlabeled_idx
    )
    
    va_set = Subset(PN_dev, val_idx)
    te_set = Subset(PN_test, np.arange(len(PN_test)))

    if verbose:
        # Print summary of dataset
        def get_y(subset):
            return [subset.dataset.targets[i]
                    for i in subset.indices]

        y_vals = np.unique(PN_dev.targets)
        row_list = []
        for splitname, dset in [('train_labeled', tr_labeled_set),
                                ('train_unlabeled', tr_unlabeled_set),
                                ('valid', va_set),
                                ('test', te_set)]:
            y_U, ct_U = np.unique(get_y(dset), return_counts=True)
            y2ct_dict = dict(zip(y_U, ct_U))
            row_dict = dict(splitname=splitname)
            for y in y_vals:
                row_dict[y] = y2ct_dict.get(y, 0)
            row_list.append(row_dict)
        df = pd.DataFrame(row_list)
        print(df.to_string(index=False))

    # Convert to DataLoaders
    labeled_loader = DataLoader(tr_labeled_set, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(tr_unlabeled_set, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False)
    return labeled_loader, unlabeled_loader, va_loader, te_loader
