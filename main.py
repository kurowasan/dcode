from train import train
import numpy as np
import argparse
from data import DataManager
import torch
import random
import os

parser = argparse.ArgumentParser('DCODE')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_type', type=str, default="maillard")  # test_interv, maillard, nonlin, lincoulped
parser.add_argument('--data_dim', type=int, default=11)
parser.add_argument('--hidden_dim', type=int, default=20)
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--n_interv', type=int, default=5)
parser.add_argument('--unseen_distr', action='store_true')
parser.add_argument('--niters', type=int, default=20000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--pooled', action='store_true')
parser.add_argument('--only_obs', action='store_true')
parser.add_argument('--exp_path', type=str, default='exp')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if args.only_obs:
    args.n_interv = 1

if args.unseen_distr:
    data_manager = DataManager(args.data_type, args.n, args.data_dim,
                               args.batch_size, args.batch_time, device,
                               args.n_interv + 1, args.exp_path)
else:
    data_manager = DataManager(args.data_type, args.n, args.data_dim,
                               args.batch_size, args.batch_time, device,
                               args.n_interv, args.exp_path)

if not os.path.exists(args.exp_path):
    os.makedirs(args.exp_path)

train(args, data_manager, device)
