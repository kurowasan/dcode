import torch
import torch.nn as nn
import torch.optim as optim
from gumbel import GumbelAdjacency
from collections import OrderedDict

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        output = self.net(y**3)
        return output


class DC(nn.Module):
    def __init__(self, num_vars, num_hidden, num_regimes):
        super().__init__()
        self.linear = []
        self.num_vars = num_vars
        self.num_hidden = num_hidden
        self.num_regimes = num_regimes
        self.interv = torch.zeros(self.num_vars)

        for v in range(num_vars):
            linear = []
            for r in range(num_regimes):
                linear.append(nn.Sequential(OrderedDict([
                                               ('lin1', nn.Linear(num_vars, num_hidden)),
                                               ('tanh', nn.Tanh()),
                                               ('lin2', nn.Linear(num_hidden, 1))
                                              ])))
            linear = nn.ModuleList(linear)
            self.linear.append(linear)

        self.linear = nn.ModuleList(self.linear)
        self.gumbel_mask = GumbelAdjacency(self.num_vars)
        self.debug = False

    def sample(self, n):
        self.mask = self.gumbel_mask(n)  # size: (bs, num_vars, num_vars)

    def interv_setting(self, interv, regime):
        self.interv = torch.zeros(self.num_vars)
        if interv is not None:
            for i in interv:
                self.interv[i] = regime
        self.debug = True

    def forward(self, t, x):
        bs = x.size(0)  # x's size: (bs, 1, num_vars)
        steps = x.size(1)

        y = []

        for v in range(self.num_vars):
            # apply mask
            if len(x.size()) == 3:
                masked_x = torch.einsum("btd,bd->btd", x, self.mask[:, v])
            elif len(x.size()) == 2:
                masked_x = torch.einsum("bd,bd->bd", x, self.mask[:, v])
            i = int(self.interv[v])

            # apply NN
            y.append(self.linear[v][i](masked_x ** 3))
        output = torch.cat(y, -1)
        if self.debug:
            self.debug = False

        return output
