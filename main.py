from train import train, train_one_equation
import argparse
from data import DataManager
import torch
import os

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_type', type=str, default="nonlin_simple")  # test_interv, maillard, nonlin
parser.add_argument('--data_dim', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--n_interv', type=int, default=1)  # 5
parser.add_argument('--unseen_distr', action='store_true')
parser.add_argument('--niters', type=int, default=20000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--adjoint', action='store_true')
# parser.add_argument('--one_equation', action='store_true')
parser.add_argument('--pooled', action='store_true')
parser.add_argument('--only_obs', action='store_true')
parser.add_argument('--exp_path', type=str, default='exp')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

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

# if args.one_equation:
#     train_one_equation(args, data_manager, device)
# else:
train(args, data_manager, device)
