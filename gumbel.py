"""
Code reused from DCDI
"""
import torch


def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)

    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft

    else:
        y = y_soft

    return y


class GumbelAdjacency(torch.nn.Module):
    """
    Random matrix M used for the mask. Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """
    def __init__(self, num_vars):
        super(GumbelAdjacency, self).__init__()
        self.num_vars = num_vars
        self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.reset_parameters()
        self.fixed = False

    def forward(self, bs, tau=1, drawhard=True):
        """Get a binary mask"""
        if not self.fixed:
            adj = gumbel_sigmoid(self.log_alpha, self.uniform, bs, tau=tau, hard=drawhard)
            return adj
        else:
            return self.fixed_ouput.repeat(bs, 1, 1)

    def get_proba(self):
        """Returns probability of getting one"""
        if not self.fixed:
            return torch.sigmoid(self.log_alpha)
        else:
            return self.fixed_ouput

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 5)
        self.fixed = False

    def fix(self, adj):
        self.fixed = True
        self.fixed_ouput = adj
