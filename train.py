import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from model import DC, DC_one_equation
from utils import RunningAverageMeter, makedirs
from plot import plot_mask, plot_mask_through_time, plot_loss, Visualization


def train(args, data, device):
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    if args.viz:
        makedirs('png')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        plt.show(block=False)

    ii = 0
    reg_coeff = 5e-4

    # num_layers = 2
    num_vars = args.data_dim
    num_hidden = args.hidden_dim

    model = DC(num_vars, num_hidden, args.n_interv)
    func = model.to(device)

    optimizer = optim.RMSprop(func.parameters(), lr=args.lr)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    loss_unseen_meter = RunningAverageMeter(0.97)
    masks = np.zeros((args.niters, num_vars, num_vars))
    losses = np.zeros((args.niters))
    losses_unseen = np.zeros((args.niters))

    viz = Visualization()

    for itr in range(1, args.niters + 1):
        k = np.random.choice(np.arange(args.n_interv), 1)[0]
        # k = 0
        # print(f"k: {k}")

        optimizer.zero_grad()
        # print(f"b_y0: {batch_y0.size()}, b_t: {batch_t.size()}, b_y: {batch_y.size()}")

        if args.pooled:
            batch_y0, batch_t, batch_y = data.get_batch(0)
            func.interv_setting([0], 0)
        else:
            batch_y0, batch_t, batch_y = data.get_batch(k)
            if k != 0:
                interv_node = data.datasets[k].interv_node
            else:
                interv_node = [0]
            func.interv_setting(interv_node, k)
        func.sample(batch_y0.shape[0])

        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        reg = reg_coeff * torch.sum(torch.sigmoid(func.gumbel_mask.log_alpha))
        reg_loss = loss + reg
        reg_loss.backward()
        optimizer.step()

        masks[itr, :, :] = torch.sigmoid(func.gumbel_mask.log_alpha).detach().numpy()
        losses[itr] = loss_meter.avg

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        # validation
        if itr % args.test_freq == 0:
            # TODO: plotting
            plot_loss(losses, itr, exp_path=args.exp_path)
            plot_mask_through_time(masks, data.datasets[0].dag.detach().numpy(), itr, exp_path=args.exp_path)
            plot_mask(torch.sigmoid(func.gumbel_mask.log_alpha).detach().numpy(),
                      data.datasets[0].dag.detach().numpy(), exp_path=args.exp_path)

            print(loss_meter.avg)
            # print(torch.sigmoid(func.gumbel_mask.log_alpha))
            ii += 1
            # with torch.no_grad():
            #     true_y0 = data.datasets[0].true_y0
            #     t = data.datasets[0].t
            #     true_y = data.datasets[0].true_y
            #     func.interv_setting([0], 0)
            #     func.sample(true_y0.shape[0])
            #     pred_y = odeint(func, true_y0, t)
            #     # pred_y = odeint(func, true_y0.unsqueeze(0), t)
            #     loss = torch.mean(torch.abs(pred_y - true_y))
            #     print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            #     # viz.plot(true_y, pred_y, t, device, func, ii)
            #     ii += 1

            #     # TODO: evaluate on unseen interv
            #     if args.unseen_distr:
            #         k = args.n_interv
            #         batch_y0, batch_t, batch_y = data.get_batch(k)
            #         func.interv_setting([0], 0)
            #         func.sample(batch_y0.shape[0])
            #         pred_y = odeint(func, batch_y0, batch_t).to(device)
            #         loss = torch.mean(torch.abs(pred_y - batch_y))
            #         print('Iter {:04d} | Total Loss {:.6f} - Unseen'.format(itr, loss.item()))

            #         loss_unseen_meter.update(loss.item())
            #         losses_unseen[ii-1] = loss_unseen_meter.avg
            #         plot_loss(losses_unseen, ii, exp_path=args.exp_path, name="_unseen")

        end = time.time()

    mask = torch.sigmoid(func.gumbel_mask.log_alpha).detach().numpy()
    gt_mask = data.datasets[0].dag.detach().numpy()
    print(np.sum(mask - gt_mask))
    # TODO: save in a file, > 0.5


def train_one_equation(args, data, device):
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint


    if args.viz:
        makedirs('png')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        plt.show(block=False)

    ii = 0
    reg_coeff = 1e-3

    # num_layers = 2
    num_vars = args.data_dim
    num_hidden = 50

    model = DC_one_equation(num_vars, num_hidden)
    func = model.to(device)

    optimizer = optim.RMSprop(func.parameters(), lr=args.lr)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        k = np.random.choice(np.arange(args.n_interv), 1)[0]

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = data.get_batch(k)
        # print(f"b_y0: {batch_y0.size()}, b_t: {batch_t.size()}, b_y: {batch_y.size()}")
        func.sample(batch_y0.shape[0])
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        __import__('ipdb').set_trace()
        loss = torch.mean(torch.abs(pred_y - batch_y))
        reg = reg_coeff * torch.sum(torch.sigmoid(func.gumbel_mask.log_alpha))
        loss += reg
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        # validation
        if itr % args.test_freq == 0:
            print(loss_meter.avg)
            print(torch.sigmoid(func.gumbel_mask.log_alpha))
            with torch.no_grad():
                true_y0 = data.datasets[0].true_y0
                t = data.datasets[0].t
                true_y = data.datasets[0].true_y
                func.sample(true_y0.shape[0])
                pred_y = odeint(func, true_y0, t)
                # pred_y = odeint(func, true_y0.unsqueeze(0), t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
