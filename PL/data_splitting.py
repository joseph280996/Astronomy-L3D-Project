import os
import shutil
import numpy as np


def lp_pl_preprocessing(dataset_path, save_path):
    # List all files
    files = os.listdir(dataset_path)

    # List with full paths
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]

    # Recursively list all files including subfolders
    True_PN_files = []
    False_PN_files = []
    for root, _, filenames in os.walk(dataset_path):
        if "True_PN" in root:
            for f in filenames:
                True_PN_files.append(os.path.join(root, f))
        if "False_PN" in root:
            for f in filenames:
                False_PN_files.append(os.path.join(root, f))

    lp_train_true_set, lp_test_true_set, pl_train_true_set, pl_test_true_set = (
        sampling_and_split(True_PN_files)
    )
    lp_train_false_set, lp_test_false_set, pl_train_false_set, pl_test_false_set = (
        sampling_and_split(False_PN_files)
    )

    if (
        not os.path.exists(os.path.join(save_path, "l3d_pn_dataset500LP"))
        or not os.path.exists(os.path.join(save_path, "l3d_pn_dataset500PL"))
        or not os.path.isdir(os.path.join(save_path, "l3d_pn_dataset500LP"))
        or not os.path.isdir(os.path.join(save_path, "l3d_pn_dataset500PL"))
    ):
        # Create destination if it doesn't exist
        save_arr = [
            ("l3d_pn_dataset500LP", "train", "True_PN", lp_train_true_set),
            ("l3d_pn_dataset500LP", "train", "False_PN", lp_train_false_set),
            ("l3d_pn_dataset500LP", "test", "True_PN", lp_test_true_set),
            ("l3d_pn_dataset500LP", "test", "False_PN", lp_test_false_set),
            ("l3d_pn_dataset500PL", "train", "True_PN", pl_train_true_set),
            ("l3d_pn_dataset500PL", "train", "False_PN", pl_train_false_set),
            ("l3d_pn_dataset500PL", "test", "True_PN", pl_test_true_set),
            ("l3d_pn_dataset500PL", "test", "False_PN", pl_test_false_set),
        ]
        save_result(save_arr, save_path)


def save_result(save_arr, save_path):
    for method, set_type, label, dataset in save_arr:
        destination_path = f"{save_path}/{method}/{set_type}/{label}"

        os.makedirs(destination_path, exist_ok=True)

        copy_files(dataset, destination_path)


def sampling_and_split(data_files):
    false_pn_arr = np.array(data_files)
    # Get random permutation of indices
    indices = np.random.permutation(500)
    # Split indices into two sets
    lp_false_set = false_pn_arr[indices[:250]]
    pl_false_set = false_pn_arr[indices[250:]]

    indices = np.random.permutation(250)
    lp_train = lp_false_set[indices[:125]]
    lp_test = lp_false_set[indices[125:]]

    indices = np.random.permutation(250)
    pl_train = pl_false_set[indices[:125]]
    pl_test = pl_false_set[indices[125:]]

    return lp_train, lp_test, pl_train, pl_test


def copy_files(file_list, destination):
    for file in file_list:
        try:
            shutil.copy2(file, destination)  # copy2 preserves metadata
            print(f"Copied {file}")
        except Exception as e:
            print(f"Error copying {file}: {e}")
