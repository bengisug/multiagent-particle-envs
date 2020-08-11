from __future__ import division
import torch
import torch.nn.functional as F

from multiagent.policies.attention.multi_head_attention import MultiHeadAttention


class AdjacencyNet(torch.nn.Module):

    def __init__(self, agent_size=10, insize=64):
        super(AdjacencyNet, self).__init__()
        self.conv1 = GraphConv()
        self.conv2 = GraphConv()
        self.adj_net = AdjDecisionNet(agent_size=agent_size, insize=insize * 3)

    def forward(self, z):
        h1, dist1 = self.conv1(z, None)
        h2, dist2 = self.conv2(h1, None)
        adj = self.adj_net(z, h1, h2)
        return adj


class GraphQNet1(torch.nn.Module):

    def __init__(self, state_size, action_size, args):
        super(GraphQNet1, self).__init__()
        self.encoder = MLP(state_size)
        self.adj_net = AdjDecisionNet(agent_size=args.nagent)
        self.conv3 = GraphConv()
        self.conv4 = GraphConv()
        self.qnet = QNet(args, action_size)
        self.last_adj = None
        self.last_a = None
        self.nagent = args.nagent
        self.device = args.device
        self.d_model = 64

    def forward(self, z, adj):
        self.last_adj = adj
        h1, dist1 = self.conv3(z, adj)
        # h2, dist2 = self.conv4(h1, adj)
        q_values = self.qnet(h1)
        return q_values, dist1, adj
        # return q_values, 0, adj

    def encode_state(self, state):
        return self.encoder(state)

    def causal_influence(self, qdist_comm, state):
        with torch.no_grad():
            adj = torch.zeros_like(self.last_adj)
            z = self.encoder(state)
            h1, dist3 = self.conv3(z, adj)
            h2, dist4 = self.conv4(h1, adj)
            q_dist = self.qnet(h2)
        return torch.nn.functional.kl_div(qdist_comm, q_dist)


class GraphConv(torch.nn.Module):

    def __init__(self, d_model=64):
        super(GraphConv, self).__init__()
        self.attention = MultiHeadAttention(d_model)
        self.o_linear = torch.nn.Linear(d_model * 2, d_model, bias=True)
        self.nonlinear = torch.nn.ReLU()

    def forward(self, x, adj):
        if x.shape[0] == 1:
            x_in = x.squeeze()
        else:
            x_in = x
        x, dist = self.attention(x, x, x, mask=adj)

        x = self.nonlinear(self.o_linear(torch.cat((x_in, x.squeeze()), dim=-1)))

        return x, dist


class QNet(torch.nn.Module):

    def __init__(self, args, action_size, insize=64):
        super(QNet, self).__init__()
        self.net = torch.nn.Sequential()
        self.net.add_module("fc_1", torch.nn.Linear(insize, action_size * args.natoms))
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.natoms = args.natoms
        self.action_size = action_size
        self.nagent = args.nagent
        self.delta_z = (self.vmax - self.vmin) / (self.natoms - 1)
        self._support = torch.arange(self.vmin, self.vmax + self.delta_z, self.delta_z)
        self.register_buffer("support", self._support)

    def forward(self, h1):
        x = h1
        # x = torch.cat((h1, h2), dim=-1)
        x = self.net(x)
        q_dist = x.view(-1, self.nagent, self.action_size, self.natoms)
        q_dist = F.softmax(q_dist, dim=-1)
        q_dist = q_dist.clamp(min=1e-4)
        return q_dist

    def expected_value(self, values):
        expected_next_value = (values * self.support).sum(dim=-1)
        return expected_next_value


class AdjDecisionNet(torch.nn.Module):

    def __init__(self, agent_size=20, insize=128 * 3):
        super(AdjDecisionNet, self).__init__()
        self.net = torch.nn.Sequential()
        self.net.add_module("fc_1", torch.nn.Linear(insize, 128))
        self.net.add_module("nonlinear_1", torch.nn.ReLU())
        self.net.add_module("fc_2", torch.nn.Linear(128, agent_size))
        self.net.add_module("activation", torch.nn.Softmax())

    def forward(self, h1, h2, h3):
        x = torch.cat((h1, h2, h3), dim=-1)
        x = self.net(x)
        x[x < 0.1] = 0
        x[x >= 0.1] = 1
        return x


class MLP(torch.nn.Module):

    def __init__(self, insize, outsize=64, nonlinear=torch.nn.ReLU, activation=torch.nn.ReLU, hidden_layer_size=1,
                 node_size=128):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential()
        self.net.add_module("fc_1", torch.nn.Linear(insize, node_size))
        self.net.add_module("nonlinear_1", nonlinear())
        for i in range(hidden_layer_size - 1):
            self.net.add_module("fc_" + str(i + 2), torch.nn.Linear(node_size, node_size))
            self.net.add_module("nonlinear_" + str(i + 2), nonlinear())
        self.net.add_module("head", torch.nn.Linear(node_size, outsize))
        self.net.add_module("activation", activation())

    def forward(self, x):
        return self.net(x)
