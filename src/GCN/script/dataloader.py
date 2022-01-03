import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import rdkit
from rdkit import Chem
import utils

def check_dimension(tensors):
    size = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            size.append(tensor.shape)
        else:
            size.append(0)
    size = np.asarray(size)
    return np.max(size, 0)


def collate_tensor(tensor, max_tensor, batch_idx):
    if isinstance(tensor, np.ndarray):
        dims = tensor.shape
        max_dims = max_tensor.shape
        slice_list = tuple([slice(0, dim) for dim in dims])
        slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
        max_tensor[tuple(slice_list)] = tensor
    elif isinstance(tensor, str):
        max_tensor[batch_idx] = tensor
    else:
        max_tensor[batch_idx] = tensor
    return max_tensor


def tensor_collate_fn(batch):

    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]

    batch_items = [it for e in batch for it in e.items()]
    total_key, total_value = list(zip(*batch_items))
    batch_size = len(batch)
    n_element = int(len(batch_items) / batch_size)
    total_key = total_key[0:n_element]

    # dim_dict
    dim_dict = dict()
    for i, k in enumerate(total_key):
        value_list = [v for j, v in enumerate(total_value)
                      if j % n_element == i]
        if isinstance(value_list[0], np.ndarray):
            dim_dict[k] = np.zeros(np.array(
                [batch_size, *check_dimension(value_list)])
            )
        elif isinstance(value_list[0], str):
            dim_dict[k] = ["" for _ in range(batch_size)]
        elif isinstance(value_list[0],list):
            print(value_list[0])
            dim_dict[k] = [[] for _ in range(batch_size)]
        else:
            dim_dict[k] = np.zeros((batch_size,))

    # ret_dict
    ret_dict = dict()
    for j in range(batch_size):
        if batch[j] == None:
            continue
        for key, max_tensor in dim_dict.items():
            value = collate_tensor(batch[j][key], max_tensor, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value
    return ret_dict


def get_dataset_dataloader(fn, batch_size=8, shuffle=True,
                           num_workers=1, length=None):
    dataset = GCNDataset(fn, length)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=shuffle,
                            collate_fn=tensor_collate_fn, drop_last=False)
    return dataset, dataloader


class GCNDataset(Dataset):
    def __init__(self, fn, length=None):
        with open(fn, "r") as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
        self.data = lines[:length]

    def __getitem__(self, idx):
        key, w_smiles,  w_HOMO, w_LUMO, w_S, w_T = self.data[idx]

        feature_dict = utils.mol_to_feature(w_smiles)

        feature_dict["key"] = key
        feature_dict["HOMO"] = float(w_HOMO)
        feature_dict["LUMO"] = float(w_LUMO)
        feature_dict["S1"] = float(w_S)
        feature_dict["T1"] = float(w_T)
        feature_dict["smiles"] = w_smiles
        return feature_dict

    def __len__(self):
        return len(self.data)
