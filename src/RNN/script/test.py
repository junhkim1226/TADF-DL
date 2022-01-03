import time
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataloader import get_dataset_dataloader,get_c_to_i
from model import RNN
from sklearn.metrics import r2_score
import utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument("--restart_file", help="file with pretrained parameter", type=str)
parser.add_argument("--num_workers", help="number of workers", type=int, default=4)
parser.add_argument('--maxlen', type=int, default=180, help='maximum length of SMILES')

parser.add_argument("--hidden_dim", help="dimension of Linear layer", type=int, default=4096)
parser.add_argument("--N_RNN_layer", help="Number of RNN layer", type=int, default=3)
parser.add_argument("--N_predictor_layer", help="Number of predictor layer", type=int, default=1)
parser.add_argument("--N_properties", help="Number of properties", type=int, default=4)
parser.add_argument("--dropout", help="dropout ratio", type=float, default=0.5)

parser.add_argument("--test_file", help="file for test data loader", type=str, default="../data/test.txt")
parser.add_argument("--test_result_file", help="file for test result", type=str, default="./output/test.txt")
parser.add_argument("--test_loss_file", help="file for test loss", type=str, default="./test_results.txt")

if __name__ == "__main__":
    # Arguments
    args = parser.parse_args()

    import pickle
    f = open('./c_to_i.pkl','rb')
    c_to_i = pickle.load(f)
    f.close()

    test_file = utils.get_abs_path(args.test_file)

    if not args.restart_file:
        utils.print_args(args)
        restart_file = None
    else:
        restart_file = utils.get_abs_path(args.restart_file)

    # Model setting
    model = RNN(args, n_char=len(c_to_i))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, restart_file)

    # Loss setting
    loss_fn = nn.L1Loss()

    test_dataset, test_dataloader = get_dataset_dataloader(test_file,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,maxlen=args.maxlen,c_to_i=c_to_i)

    with torch.no_grad():
        st = time.time()

        test_losses = []
        test_true = {}
        test_predicts = {}

        model.eval()
        for i_batch, batch in enumerate(test_dataloader):

            test_feature_dict = utils.dic_to_device(batch, device)

            x = test_feature_dict['seq'].long()
            l = test_feature_dict['length'].float()

            predict = model(x,l)

            HOMO, LUMO, S1, T1 = test_feature_dict["target"]
            property = torch.tensor(list(zip(HOMO,LUMO,S1,T1))).to(device)

            loss = loss_fn(predict, property)

            test_losses.append(loss.data.cpu().numpy())
            keys = test_feature_dict["key"]
            for idx, key in enumerate(keys):
                test_true[key] = property[idx].data.cpu().numpy()
                test_predicts[key] = predict[idx].data.cpu().numpy()

        et = time.time()

        test_loss = np.mean(np.array(test_losses))

        HOMO_true_list, LUMO_true_list = [], []
        HOMO_pred_list, LUMO_pred_list = [], []
        S1_true_list, T1_true_list = [], []
        S1_pred_list, T1_pred_list = [], []

        with open(args.test_result_file, "w") as w:
            w.write("key\ttrue\tpredict\n")
            for key in test_true.keys():
                HOMO_true, LUMO_true, S1_true, T1_true = test_true[key]
                HOMO_predict, LUMO_predict, S1_predict, T1_predict = test_predicts[key]
                HOMO_true_list.append(HOMO_true)
                LUMO_true_list.append(LUMO_true)
                S1_true_list.append(S1_true)
                T1_true_list.append(T1_true)
                HOMO_pred_list.append(HOMO_predict)
                LUMO_pred_list.append(LUMO_predict)
                S1_pred_list.append(S1_predict)
                T1_pred_list.append(T1_predict)
                w.write(f"{key}\t{HOMO_true:.4f}\t{LUMO_true:.4f}\t{S1_true:.4f}\t{T1_true:.4f}\t{HOMO_predict:.4f}\t{LUMO_predict:.4f}\t{S1_predict:.4f}\t{T1_predict:.4f}\n")
        with open(args.test_loss_file,"a+") as w:
            HOMO_MAE = loss_fn(torch.FloatTensor(HOMO_true_list), torch.FloatTensor(HOMO_pred_list))
            HOMO_r2 = r2_score(torch.FloatTensor(HOMO_true_list), torch.FloatTensor(HOMO_pred_list))
            LUMO_MAE = loss_fn(torch.FloatTensor(LUMO_true_list), torch.FloatTensor(LUMO_pred_list))
            LUMO_r2 = r2_score(torch.FloatTensor(LUMO_true_list), torch.FloatTensor(LUMO_pred_list))
            S1_MAE = loss_fn(torch.FloatTensor(S1_true_list), torch.FloatTensor(S1_pred_list))
            S1_r2 = r2_score(torch.FloatTensor(S1_true_list), torch.FloatTensor(S1_pred_list))
            T1_MAE = loss_fn(torch.FloatTensor(T1_true_list), torch.FloatTensor(T1_pred_list))
            T1_r2 = r2_score(torch.FloatTensor(T1_true_list), torch.FloatTensor(T1_pred_list))
            model_name = restart_file.split('/')[-1]
            test_loss = (HOMO_MAE+LUMO_MAE+S1_MAE+T1_MAE)/4
            w.write("%s\tHOMO_MAE\t%.4f\tHOMO_r2\t%.4f\tLUMO_MAE\t%.4f\tLUMO_r2\t%.4f\tS1_MAE\t%.4f\t\S1_r2\t%.4f\tT1_MAE\t%.4f\tT1_r2\t%.4f\tTotal\t%.4f\tTime\t%.2f\n"
                    %(model_name,
                        HOMO_MAE,HOMO_r2,LUMO_MAE,LUMO_r2,S1_MAE,S1_r2,T1_MAE,T1_r2,test_loss, et - st))
