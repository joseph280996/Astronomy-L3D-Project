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
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean_pn_RGB_3, std=stddev_pn_RGB_3),
    ]
)

IM_PREPROCESSING_FOR_VIEW = torchvision.transforms.Compose(
    [
        torchvision.transforms.CenterCrop(256),
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


def make_lp_data_loaders(
    root=os.path.abspath("../l3d_pn_dataset500LP500PL"),
    transform=DEFAULT_IM_PREPROCESSING,
    target_transform=None,
    batch_size=64,
    n_samples_per_class_trainandvalid=50,
    frac_valid=0.2,
    random_state=1234,
    verbose=True,
):
    lp_pn_dev = PNDataset(
        os.path.join(root, "LP", "train"),
        transform=transform,
        n_samples_per_class=n_samples_per_class_trainandvalid,
    )
    lp_pn_test = PNDataset(os.path.join(root, "LP", "test"), transform=transform)

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

def make_pl_data_loaders(
    root=os.path.abspath("../l3d_pn_dataset1000"),
    transform=DEFAULT_IM_PREPROCESSING,
    target_transform=None,
    batch_size=64,
    n_samples_per_class_trainandvalid=50,
    frac_valid=0.2,
    random_state=1234,
    verbose=True,
):
    pl_pn_dev = PNDataset(
        os.path.join(root, "PL", "train"),
        transform=transform,
        n_samples_per_class=n_samples_per_class_trainandvalid,
    )
    pl_pn_test = PNDataset(os.path.join(root, "PL", "test"), transform=transform)

    pl_tr_idx, pl_val_idx = train_test_split(
        np.arange(len(pl_pn_dev)),
        test_size=frac_valid,
        random_state=random_state,
        shuffle=True,
        stratify=pl_pn_dev.targets,
    )

    # Create data subsets from indices
    Subset = torch.utils.data.Subset
    pl_tr_set = Subset(pl_pn_dev, pl_tr_idx)
    pl_va_set = Subset(pl_pn_dev, pl_val_idx)
    pl_te_set = Subset(pl_pn_test, np.arange(len(pl_pn_test)))

    if verbose:
        # Print summary of dataset, in terms of counts by class for each split
        def get_y(subset):
            return [subset.dataset.targets[i] for i in subset.indices]

        y_vals = np.unique(np.union1d(get_y(pl_tr_set), get_y(pl_te_set)))
        row_list = list()
        for splitname, dset in [
            ("pl_train", pl_tr_set),
            ("pl_valid", pl_va_set),
            ("pl_test", pl_te_set),
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
    pl_tr_loader = DataLoader(pl_tr_set, batch_size=batch_size, shuffle=True)
    pl_va_loader = DataLoader(pl_va_set, batch_size=batch_size, shuffle=False)
    pl_te_loader = DataLoader(pl_te_set, batch_size=batch_size, shuffle=False)

    return pl_tr_loader, pl_va_loader, pl_te_loader

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



