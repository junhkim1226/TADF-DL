import time
import glob
import copy
import math
import subprocess
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetFormalCharge
import torch
import torch.nn as nn

##### Setting for Molecular Graph #####

SYMBOLS = ["B", "C", "N", "O", "P", "S", "X"]
DEGREE = [0, 1, 2, 3, 4, 5, 6]
IMPLICIT_VALENCE = [0, 1, 2, 3, 4, 5, 6]
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
TOTAL_NUMHS = [0, 1, 2, 3, 4]
]
N_atom_features = len(SYMBOLS)
N_extra_atom_features = len(HYBRIDIZATION + DEGREE + TOTAL_NUMHS + IMPLICIT_VALENCE)+1


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(mol, idx):
    atom = mol.GetAtomWithIdx(idx)

    symbol_onehot = one_of_k_encoding(atom.GetSymbol(), SYMBOLS)  # Symbol encoding
    degree_onehot = one_of_k_encoding(atom.GetDegree(), DEGREE)
    implicitvalence_onehot = one_of_k_encoding(atom.GetImplicitValence(),IMPLICIT_VALENCE)
    hybridization_onehot = one_of_k_encoding(atom.GetHybridization(),
                                             HYBRIDIZATION)  # Hybridization encoding
    aromatic_onehot = [atom.GetIsAromatic()]
    numH_onehot = one_of_k_encoding(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])

    at = np.array(symbol_onehot + degree_onehot + implicitvalence_onehot +
            hybridization_onehot + aromatic_onehot + numH_onehot)
    return at


def atom_to_onehot(mol):
    """Get atom features for molecule."""
    n_atoms = mol.GetNumAtoms()
    atom_features = np.zeros((n_atoms, len(SYMBOLS + DEGREE +  IMPLICIT_VALENCE +
        HYBRIDIZATION + TOTAL_NUMHS)+1))
    for i in range(n_atoms):
        atom_feature = get_atom_features(mol, i)
        if atom_feature is None:
            return None
        atom_features[i, :] = atom_feature
    atom_features = np.array(atom_features).astype(int)
    return atom_features

def mol_to_feature(smiles):
    # Molecule features in onehot
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = atom_to_onehot(mol)
    if atom_features is None:
        return None

    # Adjacency matrix
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = adj + np.eye(len(adj))

    # Validity for collate function
    atom_mask = np.ones((mol.GetNumAtoms(),))

    feature_dict = {
        "af": atom_features,
        "adj": adj,
        "atom_mask": atom_mask,
    }
    return feature_dict

##### Basic utils #####

def initialize_model(model, device, load_save_file=None):
    if load_save_file:
        print(f"=> Loading save file: {load_save_file.split('/')[-1]}")
        if device.type == "cpu":
            sdict = torch.load(load_save_file, map_location="cpu")
        else:
            sdict = torch.load(load_save_file)
        print(f"=> Dump loaded files to the model")
        model.load_state_dict(sdict)
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal_(param)
    model.to(device)
    return model

def dic_to_device(dic, device):
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value
    return dic

def print_args(args):
    name_space = []
    values = []
    for arg in vars(args):
        name_space.append(arg)
        values.append(getattr(args, arg))
    max_len_name = max([len(n) for n in name_space])
    print(f"Current Working Directory: {os.getcwd()}")
    print("####### Used Parameters #######")
    for name, value in zip(name_space, values):
        padded_name = name.ljust(max_len_name)
        print(f"\t{padded_name}: {value}")


def print_results(epoch, results, msg_length):
    line = []
    line.append(str(epoch).ljust(msg_length[0]))
    for r, ml in zip(results, msg_length[1:]):
        r = str(round(r, 3))
        line.append(r.ljust(ml))
    print("\t".join(line))


def get_abs_path(path):
    return os.path.realpath(os.path.expanduser(path))
