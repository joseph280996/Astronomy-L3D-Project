import numpy as np
import pandas as pd
import os

import torch
import torchvision

from sklearn.model_selection import train_test_split

mean_pn_RGB_3 = [0.485, 0.456, 0.406]
stddev_pn_RGB_3 = [0.229, 0.224, 0.225]

DEFAULT_IM_PREPROCESSING = torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean_pn_RGB_3, std=stddev_pn_RGB_3),
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


def make_data_loaders(
    root=os.path.abspath("../l3d_pn_dataset500LP"),
    transform=DEFAULT_IM_PREPROCESSING,
    target_transform=None,
    batch_size=64,
    n_samples_per_class_trainandvalid=50,
    frac_valid=0.2,
    random_state=1234,
    verbose=True,
):
    lp_pn_dev = PNDataset(
        os.path.join(root, "train"),
        transform=transform,
        n_samples_per_class=n_samples_per_class_trainandvalid,
    )
    lp_pn_test = PNDataset(os.path.join(root, "test"), transform=transform)

    # Stratified sampling for train and val
    lp_tr_idx, lp_val_idx = train_test_split(
        np.arange(len(lp_pn_dev)),
        test_size=frac_valid,
        random_state=random_state,
        shuffle=True,
        stratify=lp_pn_dev.targets,
    )

    # Create data subsets from indices
    Subset = torch.utils.data.Subset
    lp_tr_set = Subset(lp_pn_dev, lp_tr_idx)
    lp_va_set = Subset(lp_pn_dev, lp_val_idx)
    lp_te_set = Subset(lp_pn_test, np.arange(len(lp_pn_test)))

    if verbose:
        # Print summary of dataset, in terms of counts by class for each split
        def get_y(subset):
            return [subset.dataset.targets[i] for i in subset.indices]

        y_vals = np.unique(np.union1d(get_y(lp_tr_set), get_y(lp_te_set)))
        row_list = list()
        for splitname, dset in [
            ("lp_train", lp_tr_set),
            ("lp_valid", lp_va_set),
            ("lp_test", lp_te_set),
        ]:
            y_U, ct_U = np.unique(get_y(dset), return_counts=True)
            y2ct_dict = dict(zip(y_U, ct_U))
            row_dict = dict(splitname=splitname)
            for y in y_vals:
                row_dict[y] = y2ct_dict.get(y, 0)
            row_list.append(row_dict)
        df = pd.DataFrame(row_list)
        print(df.to_string(index=False))

    # Convert to DataLoaders
    DataLoader = torch.utils.data.DataLoader
    lp_tr_loader = DataLoader(lp_tr_set, batch_size=batch_size, shuffle=True)
    lp_va_loader = DataLoader(lp_va_set, batch_size=batch_size, shuffle=False)
    lp_te_loader = DataLoader(lp_te_set, batch_size=batch_size, shuffle=False)

    return lp_tr_loader, lp_va_loader, lp_te_loader
