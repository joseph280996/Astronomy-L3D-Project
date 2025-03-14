import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split

# Mean/stddev of R/G/B image channels for ImageNet
mean_inet_RGB_3 = [0.485, 0.456, 0.406]
stddev_inet_RGB_3 = [0.229, 0.224, 0.225]

# Enhanced transforms for labeled data
labeled_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_inet_RGB_3, std=stddev_inet_RGB_3),
])

# Enhanced transforms for unlabeled data (stronger augmentations)
unlabeled_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_inet_RGB_3, std=stddev_inet_RGB_3),
])

# For visualization
IM_PREPROCESSING_FOR_VIEW = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),  
])

class PNDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, 
            transform=None, 
            target_transform=None,
            n_samples_per_class=None):
        super().__init__(root,
            transform=transform, target_transform=target_transform)
        self.n_samples_per_class = n_samples_per_class

        if self.n_samples_per_class is not None:
            self._filter_samples()

    def transform_for_viz(self, x):
        return IM_PREPROCESSING_FOR_VIEW(x)

    def _filter_samples(self):
        class_counts = {}
        filtered_samples = []
        
        # Sort samples to ensure deterministic sampling
        sorted_samples = sorted(self.samples, key=lambda x: x[0])
        
        for sample, target in sorted_samples:
            if target not in class_counts:
                class_counts[target] = 0

            if class_counts[target] < self.n_samples_per_class:
                filtered_samples.append((sample, target))
                class_counts[target] += 1

        self.samples = filtered_samples
        self.targets = [target for _, target in filtered_samples]

def make_PN_data_loaders_with_unlabeled(
        root=os.path.abspath('.'),
        batch_size=32,
        n_samples_per_class_trainandvalid=250,
        frac_valid=0.2,  # Increased validation fraction
        frac_unlabeled=0.5,
        random_state=42,
        verbose=True):
    
    # Load datasets with appropriate transforms
    PN_dev = PNDataset(os.path.join(root, 'train'))
    PN_test = PNDataset(os.path.join(root, 'test'))
    
    if verbose:
        print("Class mapping:", PN_dev.class_to_idx)
        print("Total training samples:", len(PN_dev))
        print("Total test samples:", len(PN_test))

    # Stratified split for train and validation
    tr_idx, val_idx = train_test_split(
        np.arange(len(PN_dev)),
        test_size=frac_valid,
        random_state=random_state,
        shuffle=True,
        stratify=PN_dev.targets
    )

    # Split training data into labeled and unlabeled
    labeled_size = int(len(tr_idx) * (1 - frac_unlabeled))
    unlabeled_size = len(tr_idx) - labeled_size

    # Ensure deterministic split
    generator = torch.Generator().manual_seed(random_state)
    labeled_idx, unlabeled_idx = random_split(
        tr_idx, 
        [labeled_size, unlabeled_size],
        generator=generator
    )

    # Create datasets with appropriate transforms
    tr_labeled_set = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=labeled_transform),
        labeled_idx
    )
    
    tr_unlabeled_set = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=unlabeled_transform),
        unlabeled_idx
    )
    
    va_set = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=labeled_transform),
        val_idx
    )
    
    te_set = PN_test

    if verbose:
        # Print dataset statistics
        def get_y(subset):
            return [subset.dataset.targets[i] for i in subset.indices]

        y_vals = np.unique(PN_dev.targets)
        row_list = []
        for splitname, dset in [
            ('train_labeled', tr_labeled_set),
            ('train_unlabeled', tr_unlabeled_set),
            ('valid', va_set),
            ('test', te_set)
        ]:
            y_U, ct_U = np.unique(get_y(dset), return_counts=True)
            y2ct_dict = dict(zip(y_U, ct_U))
            row_dict = dict(splitname=splitname)
            for y in y_vals:
                row_dict[y] = y2ct_dict.get(y, 0)
            row_list.append(row_dict)
        df = pd.DataFrame(row_list)
        print("\nDataset split statistics:")
        print(df.to_string(index=False))

    # Create DataLoaders with appropriate batch sizes and sampling
    labeled_loader = DataLoader(
        tr_labeled_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    unlabeled_loader = DataLoader(
        tr_unlabeled_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    va_loader = DataLoader(
        va_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    te_loader = DataLoader(
        te_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return labeled_loader, unlabeled_loader, va_loader, te_loader