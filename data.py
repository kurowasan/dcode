import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from plot import plot_trajectories

class Lambda(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = torch.nn.Parameter(A)

    def forward(self, t, y):
        output = torch.mm(y**3, self.A)
        return output

class NonLinearModel(nn.Module):
    def __init__(self, interv):
        super().__init__()
        ks = [0, 0.01, 0.00509, 0.0047, 0.011, 0.00712, 0.00439, 0.018, 0.011134, 0.014359, 0.00015, 0.12514]
        self.k = torch.nn.Parameter(torch.FloatTensor(ks))
        self.s = torch.tensor([[1, 1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 1, 0],
                               [1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 1]])
        self.init_cond = torch.FloatTensor([[1, 2, 3, 4, 5, 6]])

        if interv > 1:
            self.init_cond[0, 0] = torch.rand(1) * 10
            self.init_cond[0, 1] = torch.rand(1) * 10
            self.init_cond[0, 1] = torch.rand(1) * 10
            self.init_cond[0, 1] = torch.rand(1) * 10
            self.init_cond[0, 1] = torch.rand(1) * 10
            self.init_cond[0, 1] = torch.rand(1) * 10

    def forward(self, t, y):
        k = self.k
        output = torch.zeros((y.size(0), y.size(1)))
        output[:, 0] = k[1] * y[:, 0] * y[:, 1]
        output[:, 1] = k[2] * y[:, 0] + k[3] * y[:, 4]
        output[:, 2] = np.sin(k[3] * y[:, 0] + y[:, 1])
        output[:, 3] = (k[4] * y[:, 1] + y[:, 4]) ** 2
        output[:, 4] = np.cos(k[5] * y[:,2])
        output[:, 5] = np.cos(k[6] * y[:, 3] * y[:, 5])

        return output

class BioModel(nn.Module):
    """ Biomodel 52 from XXX """
    def __init__(self, interv):
        super().__init__()
        ks = [0, 0.01, 0.00509, 0.00047, 0.0011, 0.00712, 0.00439, 0.00018, 0.11134, 0.14359, 0.00015, 0.12514]
        self.k = torch.nn.Parameter(torch.FloatTensor(ks))
        self.s = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        self.init_cond = torch.FloatTensor([[160, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0]])
        self.interv = interv

        if interv == 1 or interv == 2:
            self.init_cond[0, 0] = torch.rand(1) * 5 * 160
            self.init_cond[0, -2] = torch.rand(1) * 5 * 15
            print(self.init_cond)
        elif interv == 3:
            self.interv_node = [9]
        elif interv == 4:
            self.interv_node = [4]
        elif interv > 4:
            self.init_cond[0, 0] = torch.rand(1) * 5 * 160
            self.init_cond[0, 1] = torch.rand(1) * 5 * 160
            self.init_cond[0, 2] = torch.rand(1) * 5 * 160
            self.init_cond[0, 3] = torch.rand(1) * 5 * 160
            self.init_cond[0, 4] = torch.rand(1) * 5 * 160
            self.init_cond[0, 5] = torch.rand(1) * 5 * 160
            self.init_cond[0, 6] = torch.rand(1) * 5 * 160
            self.init_cond[0, 7] = torch.rand(1) * 5 * 160
            self.init_cond[0, 8] = torch.rand(1) * 5 * 160
            self.init_cond[0, 9] = torch.rand(1) * 5 * 160
            self.init_cond[0, 10] = torch.rand(1) * 5 * 160
            print(self.init_cond)

    def forward(self, t, y):
        k = self.k

        output = torch.zeros((y.size(0), y.size(1)))
        glu = y[:, 0]
        fru = y[:, 1]
        formic = y[:, 2]
        triose = y[:, 3]
        acetic = y[:, 4]
        cn = y[:, 5]
        amadori = y[:, 6]
        amp = y[:, 7]
        c5 = y[:, 8]
        lys = y[:, 9]
        melanoidin = y[:, 10]

        # ODE of the reactions
        output[:, 0] = -(k[1] + k[3]) * glu + k[2] * fru + k[7] * glu * lys
        output[:, 1] = k[1] * glu - (k[2] + k[4] + k[5]) * fru - k[10] * fru * lys
        output[:, 2] = k[3] * glu + k[4] * fru
        output[:, 3] = 2 * k[5] * fru - k[6] * triose
        output[:, 4] = k[6] * triose + k[8] * amadori
        output[:, 5] = k[6] * triose
        output[:, 6] = -(k[8] + k[9]) * amadori + k[7] * glu * lys
        output[:, 7] = k[9] * amadori - k[11] * amp + k[10] * fru * lys
        output[:, 8] = k[3] * glu + k[4] * fru
        output[:, 9] = k[8] * amadori - k[7] * glu * lys - k[10] * fru * lys
        output[:, 10] = k[11] * amp

        if self.interv == 3:
            output[:, 9] = k[8] * amadori
        elif self.interv == 4:
            output[:, 4] = k[6] * triose

        # TODO: add noise!

        return output


def sample_dag(nodes, expected_density):
    """ Create the structure of the graph """

    adjacency_matrix = torch.zeros((nodes, nodes))
    nb_edges = expected_density * nodes
    prob_connection = 2 * nb_edges/(nodes**2 - nodes)
    causal_order = np.random.permutation(np.arange(nodes))

    for i in range(nodes - 1):
        node = causal_order[i]
        possible_parents = causal_order[(i+1):]
        num_parents = np.random.binomial(n=nodes - i - 1, p=prob_connection)
        parents = np.random.choice(possible_parents, size=num_parents, replace=False)
        adjacency_matrix[parents,node] = 1

    # causal_order = causal_order[::-1]

    return adjacency_matrix


def sample_binary_matrix(d, prob=0.5):
    p = torch.ones((d, d)) * prob
    return torch.bernoulli(p)


class Dataset:
    def __init__(self, data_type, n, d, batch_size, batch_time, device, dag,
                 i_interv, exp_path, obs_model=None):
        self.data_type = data_type
        self.i_interv = i_interv
        self.n = n
        self.d = d
        self.batch_size = batch_size
        self.batch_time = batch_time
        self.device = device
        self.interv_node = None
        self.dag = dag
        self.data_type = data_type
        self.obs_model = obs_model
        self.device = device


        if self.data_type == "test_interv":
            self.t = torch.linspace(0., 25., n).to(device)
            blow_up = True
            while(blow_up):
                self.generate_data()

                lambda_f = Lambda(self.true_A)
                with torch.no_grad():
                    self.true_y = odeint(lambda_f, self.true_y0, self.t, method='dopri5')

                if torch.max(torch.norm(self.true_y, dim=2)) < 10000:
                    print(torch.max(torch.norm(self.true_y, dim=2)))
                    blow_up = False
                else:
                    print(torch.max(torch.norm(self.true_y, dim=2)))
                    print("Blow up!")

        elif self.data_type == "maillard":
            self.t = torch.linspace(0., 100., n).to(device)
            # self.generate_data()

            reactants = ["glu", "fru", "formic", "triose", "acetic", "cn",
                         "amadori", "amp", "c5", "lys", "melanoidin"]
            bio_model = BioModel(i_interv)
            self.d = 11
            self.true_y0 = bio_model.init_cond
            self.dag = bio_model.s
            with torch.no_grad():
                self.true_y = odeint(bio_model, self.true_y0, self.t, method='dopri5')
            plot_trajectories(self.t, self.true_y, reactants, exp_path, f"_{i_interv}")
        elif self.data_type == "nonlin":
            self.t = torch.linspace(0., 25., n).to(device)
            nonlin_model = NonLinearModel(i_interv)
            self.d = 6
            self.true_y0 = nonlin_model.init_cond
            self.dag = nonlin_model.s
            with torch.no_grad():
                self.true_y = odeint(nonlin_model, self.true_y0, self.t, method='dopri5')


    def generate_data(self):
        if self.data_type == "original":
            self.d = 2
            self.true_y0 = torch.tensor([[2., 0.]]).to(self.device)
            self.true_A = torch.tensor([[-0.1, 1.5], [-1.5, -0.1]]).to(self.device)
        elif self.data_type == "test_interv" and self.i_interv == 0:
            self.d = 4
            self.true_y0 = torch.tensor([[2., 0., 1., 1.]]).to(self.device)
            self.true_A = torch.tensor([[-0.1, 1.5, 0, 0],
                                        [-1.5, -0.1, 0, 0],
                                        [0, 0, -0.2, 2],
                                        [0, 0, -2, -0.2]]).to(self.device)
            self.dag = torch.tensor([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]])
        elif self.data_type == "test_interv" and self.i_interv == 1:
            self.d = 4
            self.true_y0 = torch.tensor([[2., 0., 1., 1.]]).to(self.device)
            self.true_A = torch.tensor([[-0.3, 2, 0, 0],
                                        [-1.5, -0.1, 0, 0],
                                        [0, 0, -0.2, 2],
                                        [0, 0, -2, -0.2]]).to(self.device)
            self.dag = torch.tensor([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]])
            self.interv_node = [0]
        elif self.data_type == "test_interv" and self.i_interv == 2:
            self.d = 4
            self.true_y0 = torch.tensor([[2., 0., 1., 1.]]).to(self.device)
            self.true_A = torch.tensor([[-0.1, 1.5, 0, 0],
                                        [-1, -0.5, 0, 0],
                                        [0, 0, -0.2, 2],
                                        [0, 0, -2, -0.2]]).to(self.device)
            self.dag = torch.tensor([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]])
            self.interv_node = [1]
        elif self.data_type == "test_interv" and self.i_interv == 3:
            self.d = 4
            self.true_y0 = torch.tensor([[2., 0., 1., 1.]]).to(self.device)
            self.true_A = torch.tensor([[-0.1, 1.5, 0, 0],
                                        [-1.5, -0.1, 0, 0],
                                        [0, 0, -0.6, 1.5],
                                        [0, 0, -2, -0.2]]).to(self.device)
            self.dag = torch.tensor([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]])
            self.interv_node = [2]
        elif self.data_type == "test_interv" and self.i_interv == 4:
            self.d = 4
            self.true_y0 = torch.tensor([[2., 0., 1., 1.]]).to(self.device)
            self.true_A = torch.tensor([[-0.01, -0.01, 0, 0],
                                        [-0.01, 0.01, 0, 0],
                                        [0, 0, -0.2, 2],
                                        [0, 0, -2, -0.2]]).to(self.device)
            self.dag = torch.tensor([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]])
            self.interv_node = [0, 1]
        elif self.data_type == "test_interv" and self.i_interv == 5:
            self.d = 4
            self.true_y0 = torch.tensor([[2., 0., 1., 1.]]).to(self.device)
            self.true_A = torch.tensor([[-0.1, 1.5, 0, 0],
                                        [-1.5, -0.1, 0, 0],
                                        [0, 0, -0.9, 1.2],
                                        [0, 0, -2, -0.2]]).to(self.device)
            # self.true_A = torch.tensor([[-0.01, -0.01, 0, 0],
            #                             [-0.01, 0.01, 0, 0],
            #                             [0, 0, -0.2, 2],
            #                             [0, 0, -2, -0.2]]).to(self.device)
            self.dag = torch.tensor([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]])
            # self.interv_node = [0, 1]
            self.interv_node = [2]
        else:
            if self.obs_model is None:
                self.true_y0 = torch.rand((1, self.d)).to(self.device)
                true_A = torch.normal(0, 1, size=(self.d, self.d))
                self.true_A = true_A * self.dag
            else:
                # self.true_y0 = torch.rand((1, self.d)).to(self.device)
                self.true_y0 = torch.clone(self.obs_model.true_y0)
                true_A = torch.clone(self.obs_model.true_A)
                rand_i = np.random.choice(np.arange(self.d), 1)
                # TODO: check if should change column or row...
                true_A[rand_i, :] = torch.normal(0, 1, size=(1, self.d))
                self.interv_node = rand_i
                self.true_A = true_A * self.obs_model.dag

            print(self.true_A)
            print(self.dag)
            print(self.true_y0)


    def get_batch(self):
        s = torch.from_numpy(np.random.choice(np.arange(self.n - self.batch_time, dtype=np.int64), self.batch_size, replace=False))
        batch_y0 = self.true_y[s]  # (M, D)
        batch_t = self.t[:self.batch_time]  # (T)
        batch_y = torch.stack([self.true_y[s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
        # print(batch_y0.size(), batch_t.size(), batch_y.size())
        return batch_y0.to(self.device), batch_t.to(self.device), batch_y.to(self.device)


class DataManager:
    def __init__(self, data_type, n, d, batch_size, batch_time, device,
                 n_interv=1, exp_path=""):
        self.datasets = []
        self.d = d
        self.exp_path = exp_path

        if data_type not in ["original", "test", "test_interv"]:
            # self.dag = sample_dag(d, 1.5)
            self.dag = sample_binary_matrix(d, 0.4)
        else:
            self.dag = None

        self.datasets.append(Dataset(data_type, n, self.d, batch_size, batch_time,
                                     device, self.dag, 0, exp_path))

        for i in range(1, n_interv):
            self.datasets.append(Dataset(data_type, n, self.d, batch_size, batch_time,
                                         device, self.dag, i, exp_path, self.datasets[0]))

    def get_batch(self, k=0):
        return self.datasets[k].get_batch()
