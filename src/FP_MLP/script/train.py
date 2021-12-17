import time
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataloader import get_dataset_dataloader
from model import FpMLP
import utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument("--restart_file", help="file with pretrained parameter", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
parser.add_argument("--lr_decay", help="learning rate decay ratio", type=float, default=1)
parser.add_argument("--num_workers", help="number of workers", type=int, default=4)

parser.add_argument("--fp_dim", help="dimension of fingerprint", type=int, default=1024)
parser.add_argument("--hidden_dim", help="dimension of Linear layer", type=int, default=4096)
parser.add_argument("--N_MLP_layer", help="Number of predict layer", type=int, default=3)
parser.add_argument("--N_predictor_layer", help="Number of predictor layer", type=int, default=1)
parser.add_argument("--N_properties", help="Number of properties", type=int, default=4)
parser.add_argument("--dropout", help="dropout ratio", type=float, default=0.5)

parser.add_argument("--train_file", help="file for train data loader", type=str, default="../data/train.txt")
parser.add_argument("--val_file", help="file for validation data loader", type=str, default="../data/val.txt")
parser.add_argument("--num_epochs", help="number of training epochs", type=int, default=1500)
parser.add_argument("--save_dir", help="directory to save model parameters", type=str, default="./results")
parser.add_argument("--save_every", help="how frequently save model parameters", type=int, default=10)
parser.add_argument("--train_result_file", help="file for train result", type=str, default="./output/train.txt")
parser.add_argument("--val_result_file", help="file for validation result", type=str, default="./output/val.txt")

if __name__ == "__main__":
    # Arguments
    args = parser.parse_args()

    train_file = utils.get_abs_path(args.train_file)
    val_file = utils.get_abs_path(args.val_file)
    save_dir = utils.get_abs_path(args.save_dir)

    if not args.restart_file:
        utils.print_args(args)
        restart_file = None
    else:
        restart_file = utils.get_abs_path(args.restart_file)

    save_dir = utils.get_abs_path(args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Model setting
    model = FpMLP(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, restart_file)

    # Loss setting
    loss_fn = nn.L1Loss()

    # Optimizer setting
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset, train_dataloader = get_dataset_dataloader(
        train_file, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset, val_dataloader = get_dataset_dataloader(
        val_file, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    msg = "Epoch\tTrain_Loss\tVal_Loss\t\tTime"
    msg_length = [len(m) for m in msg.split("\t")]
    print(msg)

    for epoch in range(args.num_epochs):
        st = time.time()

        train_losses = []
        train_true = {}
        train_predicts = {}

        model.train()
        for i_batch, batch in enumerate(train_dataloader):

            model.zero_grad()
            optimizer.zero_grad()
            train_feature_dict = utils.dic_to_device(batch, device)

            x = train_feature_dict['fp'].float()

            predict = model(x)

            HOMO, LUMO, S1, T1 = train_feature_dict["target"]
            property = torch.tensor(list(zip(HOMO,LUMO,S1,T1))).to(device)

            loss = loss_fn(predict, property)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.data.cpu().numpy())

            keys = train_feature_dict["key"]
            for idx, key in enumerate(keys):
                train_true[key] = property[idx].data.cpu().numpy()
                train_predicts[key] = predict[idx].data.cpu().numpy()

        val_losses = []
        val_true = {}
        val_predicts = {}

        model.eval()
        for i_batch, batch in enumerate(val_dataloader):
            val_feature_dict = utils.dic_to_device(batch, device)

            x = val_feature_dict['fp'].float()

            predict = model(x)

            HOMO, LUMO, S1, T1 = val_feature_dict["target"]
            property = torch.tensor(list(zip(HOMO,LUMO,S1,T1))).to(device)

            loss = loss_fn(predict, property)

            val_losses.append(loss.data.cpu().numpy())

            keys = val_feature_dict["key"]
            for idx, key in enumerate(keys):
                val_true[key] = property[idx].data.cpu().numpy()
                val_predicts[key] = predict[idx].data.cpu().numpy()

        et = time.time()
        train_loss = np.mean(np.array(train_losses))
        val_loss = np.mean(np.array(val_losses))

        with open(args.train_result_file, "w") as w:
            w.write("key\ttrue\tpredict\n")
            for key in train_true.keys():
                HOMO_true, LUMO_true, S1_true,T1_true = train_true[key]
                HOMO_predict, LUMO_predict, S1_predict, T1_predict = train_predicts[key]
                w.write(f"{key}\t{HOMO_true:.3f}\t{LUMO_true:.3f}\t{S1_true:.3f}\t{T1_true:.3f}\t{HOMO_predict:.3f}\t{LUMO_predict:.3f}\t{S1_predict:.3f}\t{T1_predict:.3f}\n")
        with open(args.val_result_file, "w") as w:
            w.write("key\ttrue\tpredict\n")
            for key in val_true.keys():
                HOMO_true, LUMO_true, S1_true,T1_true = val_true[key]
                HOMO_predict, LUMO_predict, S1_predict, T1_predict = val_predicts[key]
                w.write(f"{key}\t{HOMO_true:.3f}\t{LUMO_true:.3f}\t{S1_true:.3f}\t{T1_true:.3f}\t{HOMO_predict:.3f}\t{LUMO_predict:.3f}\t{S1_predict:.3f}\t{T1_predict:.3f}\n")

        epoch_result = [train_loss, val_loss, et - st]
        utils.print_results(epoch, epoch_result, msg_length)

        name = os.path.join(save_dir, f"save_{epoch}.pt")
        if epoch % args.save_every == 0:
            #torch.save(model.module.state_dict(), name)
            torch.save(model.state_dict(), name)

        lr = args.lr * ((args.lr_decay) ** epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
