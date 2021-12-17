import os

import numpy as np
import torch
import torch.nn as nn

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
