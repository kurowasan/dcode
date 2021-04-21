import os

import torch
import matplotlib

# To avoid displaying the figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import makedirs


def plot_mask_through_time(mask, gt_adjacency, num_iter, exp_path,
                            name="mask_weight"):
    num_vars = mask.shape[1]
    max_value = 0
    fig, ax1 = plt.subplots()

    # Plot weight of incorrect edges
    for i in range(num_vars):
        for j in range(num_vars):
            if gt_adjacency[i, j]:
                continue
            else:
                color = 'r'
            y = mask[1:num_iter, i, j]
            ax1.plot(range(1, num_iter), y, color, linewidth=1)
            if len(y) > 0: max_value = max(max_value, np.max(y))

    # Plot weight of correct edges
    for i in range(num_vars):
        for j in range(num_vars):
            if gt_adjacency[i, j]:
                color = 'g'
            else:
                continue
            y = mask[1:num_iter, i, j]
            ax1.plot(range(1, num_iter), y, color, linewidth=1)
            if len(y) > 0: max_value = max(max_value, np.max(y))

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(name)

    fig.tight_layout()
    fig.savefig(os.path.join(exp_path, name + '.png'))
    fig.clf()


def plot_mask(mask, gt_mask, exp_path, name=''):
    """Plot the probability of the learned mask and compare it to the ground
    truth"""
    plt.clf()
    f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
    sns.heatmap(mask, ax=ax1, cbar=False, vmin=-1, vmax=1, cmap="Blues_r", xticklabels=False, yticklabels=False)
    sns.heatmap(gt_mask, ax=ax2, cbar=False, vmin=-1, vmax=1, cmap="Blues_r", xticklabels=False, yticklabels=False)

    ax1.set_title("Learned")
    ax2.set_title("Ground truth")

    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')

    plt.savefig(os.path.join(exp_path, 'mask' + name + '.png'))
    plt.clf()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_loss(loss, reg, num_iter, exp_path, name=""):
    if num_iter > 50:
        plt.clf()
        avg_loss = moving_average(loss[:num_iter], 40)
        avg_reg = moving_average(reg[:num_iter], 40)
        plt.plot(range(len(avg_loss)), avg_loss)
        plt.plot(range(len(avg_reg)), avg_reg)
        plt.savefig(os.path.join(exp_path, 'loss' + name + '.png'))


def plot_trajectories(t, data, traj_name, exp_path, name):
    plt.clf()
    for i in range(data.size(2)):
        plt.plot(t, data[:, 0, i], label=traj_name[i])
    plt.legend()
    plt.savefig(os.path.join(exp_path, 'trajectories' + name + '.png'))


class Visualization:
    """ Code taken from ode_demo.py of torchdiffeq """
    def __init__(self):
        makedirs('png')
        self.fig = plt.figure(figsize=(12, 4), facecolor='white')
        self.ax_traj = self.fig.add_subplot(131, frameon=False)
        self.ax_phase = self.fig.add_subplot(132, frameon=False)
        self.ax_vecfield = self.fig.add_subplot(133, frameon=False)
        plt.show(block=False)

    def plot(self, true_y, pred_y, t, device, odefunc, itr):

        print(true_y.size())
        print(pred_y.size())
        self.ax_traj.cla()
        self.ax_traj.set_title('Trajectories')
        self.ax_traj.set_xlabel('t')
        self.ax_traj.set_ylabel('x,y')
        self.ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        self.ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        self.ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        self.ax_traj.set_ylim(-2, 2)
        self.ax_traj.legend()

        self.ax_phase.cla()
        self.ax_phase.set_title('Phase Portrait')
        self.ax_phase.set_xlabel('x')
        self.ax_phase.set_ylabel('y')
        self.ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        self.ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        self.ax_phase.set_xlim(-2, 2)
        self.ax_phase.set_ylim(-2, 2)

        self.ax_vecfield.cla()
        self.ax_vecfield.set_title('Learned Vector Field')
        self.ax_vecfield.set_xlabel('x')
        self.ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        data = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))
        pad = torch.zeros(21 * 21 , 2)
        data = torch.cat((data, pad), dim=1)
        dydt = odefunc(0, data).to(device).cpu().detach().numpy()
        dydt = dydt[:, 0:2]
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        self.ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        self.ax_vecfield.set_xlim(-2, 2)
        self.ax_vecfield.set_ylim(-2, 2)

        self.fig.tight_layout()
        self.fig.savefig('png/{:03d}'.format(itr))
