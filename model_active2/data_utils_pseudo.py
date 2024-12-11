import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms
import torch
import torchvision

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

# Mean/stddev of R/G/B image channels for ImageNet
mean_inet_RGB_3 = [0.485, 0.456, 0.406]
stddev_inet_RGB_3 = [0.229, 0.224, 0.225]

#color_transforms = transforms.Compose([
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Change brightness, contrast, saturation, and hue
#    transforms.RandomHorizontalFlip(),  # You can add other augmentations like flipping, etc.
#    transforms.ToTensor(),  # Convert image to tensor
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization for pretrained models
#])


DEFAULT_IM_PREPROCESSING = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),  
    #torchvision.transforms.Normalize(mean=mean_inet_RGB_3, std=stddev_inet_RGB_3),
])


#DEFAULT_IM_PREPROCESSING = transforms.Compose([
#    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color-specific augmentations
#    transforms.CenterCrop(224),  # Crop the image to 224x224
#    transforms.ToTensor(),  # Convert image to tensor
#    transforms.Normalize(mean=mean_inet_RGB_3, std=stddev_inet_RGB_3),  # Normalize the image
#])

# For labeled data
labeled_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=mean_inet_RGB_3, std=stddev_inet_RGB_3),
])

# For unlabeled data (stronger augmentations)
unlabeled_color_jitter = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

unlabeled_random_crop = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomRotation(15),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# For visualization to show humans, don't do the normalization
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

        for sample, target in self.samples:
            if target not in class_counts:
                class_counts[target] = 0

            if class_counts[target] < self.n_samples_per_class:
                filtered_samples.append((sample, target))
                class_counts[target] += 1

        self.samples = filtered_samples
        self.targets = [target for _, target in filtered_samples]

def make_PN_data_loaders_with_unlabeled(
        root=os.path.abspath('.'),
        transform=DEFAULT_IM_PREPROCESSING,
        batch_size=32,
        frac_valid=0.2,
        frac_unlabeled=0.5,  # Fraction of training set to use as unlabeled
        random_state=23,
        verbose=True):
    PN_dev = PNDataset(os.path.join(root, 'train'), transform=transform, n_samples_per_class=425)
    
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

    # Subsets for training, validation, and test
    # tr_labeled_set = Subset(PN_dev, labeled_idx)
    # tr_unlabeled_set = Subset(PN_dev, unlabeled_idx)
    
    # Subset for labeled data
    #tr_labeled_set = Subset(
    #    PNDataset(root=os.path.join(root, 'train'), transform=labeled_transform), labeled_idx
    #)
    
    # Subset for labeled data
    tr_labeled_set = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=labeled_transform), labeled_idx
    )

    # Subset for unlabeled data
    tr_unlabeled_set1 = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=unlabeled_color_jitter), unlabeled_idx
    )

    tr_unlabeled_set2 = Subset(
        PNDataset(root=os.path.join(root, 'train'), transform=unlabeled_random_crop), unlabeled_idx
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
                                ('train_unlabeled1', tr_unlabeled_set1),
                                ('train_unlabeled2', tr_unlabeled_set2),
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
    unlabeled_loader1 = DataLoader(tr_unlabeled_set1, batch_size=batch_size, shuffle=True)
    unlabeled_loader2 = DataLoader(tr_unlabeled_set2, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=batch_size, shuffle=False)
    return labeled_loader, unlabeled_loader1, unlabeled_loader2, va_loader, te_loader
