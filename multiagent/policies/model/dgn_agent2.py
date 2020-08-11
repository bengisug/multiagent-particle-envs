import torch, random
import numpy as np
from copy import deepcopy

from multiagent.policies.model.off_policy_agent import OffPolicyAgent


class DGNAgent(OffPolicyAgent):

    def __init__(self, nact, nagent, dgnet, lr, reg_coef, start_epsilon, end_epsilon, epsilon_decay, transition_type,
                 buffer_size, batch_size, tau, gamma, grad_clip, device, handle=0):
        super(DGNAgent, self).__init__(transition_type, buffer_size, batch_size, tau, gamma, grad_clip, device)
        self.nact = nact
        self.nagent = nagent
        self.dgnet = dgnet
        self.target_dgnet = deepcopy(dgnet)
        self.opt = torch.optim.Adam(self.dgnet.parameters(), lr=lr)

        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
        self.reg_coef = reg_coef

        self.to(self.device)
        self.handle = handle

    def act(self, observation, adjacency_list, epsilon):
        r = random.random()
        actions, causal_influence, adj = (self.act_greedy(observation, adjacency_list)) if r < (1.0 - epsilon) else (
        np.array(
            [[random.choice(range(self.nact))] for _ in range(self.nagent)], dtype=np.int32), np.nan, np.nan)
        return actions, causal_influence, adj

    def act_greedy(self, observation, adjacency_list):
        self.eval()
        with torch.no_grad():
            q_dist, dist, adj = self.dgnet(observation, adjacency_list)
            q_values = self.dgnet.qnet.expected_value(q_dist)
            actions = torch.max(q_values, dim=-1).indices.squeeze().unsqueeze(dim=-1)
            causal_influence = self.dgnet.causal_influence(q_dist, observation)
        return actions.to("cpu").numpy().astype(np.int32), causal_influence, adj

    def infer_edge(self, state):
        adj = []
        for j in range(len(state)):
            neighbor_count = 3
            f = [[(state[r][-2] - state[j][-2]) ** 2 + (state[r][-1] - state[j][-1]) ** 2, r] for r in
                 range(len(state))]
            f.sort(key=lambda x: x[0])
            y = [f[r][1] for r in range(neighbor_count + 1)]
            y = np.eye(len(state))[y]
            y = y.sum(0)
            adj.append(y)
        return adj

    def _td_loss(self, batch):
        batch_size = batch.states.size()[0]
        with torch.no_grad():
            target_qvalue_dist, _, _ = self.target_dgnet(batch.next_states, batch.adjs)
            target_qvalues = self.dgnet.qnet.expected_value(target_qvalue_dist)
            target_actions = torch.max(target_qvalues, dim=-1).indices.view(-1)
            target_best_dist = target_qvalue_dist.view(-1, self.nact, self.dgnet.qnet.natoms)[
                range(batch_size * self.nagent), target_actions.view(-1)]

            _, next_dist, _ = self.dgnet(batch.next_states, batch.adjs)

        projected_dist = self.project_dist(target_best_dist, batch, self.gamma)

        qvalue_dist, dist, _ = self.dgnet(batch.states, batch.adjs)
        causal_influence = self.target_dgnet.causal_influence(qvalue_dist, batch.states)
        q_best_dist = qvalue_dist.view(-1, self.nact, self.dgnet.qnet.natoms)[
            range(batch_size * self.nagent), batch.actions.squeeze().view(-1).long()]
        logprob = torch.log(q_best_dist)

        td_loss = -logprob * projected_dist
        td_loss = td_loss.sum(dim=-1).mean()

        div = torch.nn.functional.kl_div(dist, next_dist)

        td_loss += self.reg_coef * div
        # td_loss -= causal_influence

        return td_loss

    def update(self):
        self.train()
        batch = self.transition_buffer.sample(self.batch_size)
        batch = self._batchtotorch(batch)

        self.opt.zero_grad()
        td_loss = self._td_loss(batch)
        td_loss.backward()

        if self.grad_clip:
            self._clip_grad(self.dgnet.parameters())
        self.opt.step()

        self.update_target()

        return td_loss.item()

    def update_target(self):
        self._update_target(self.dgnet, self.target_dgnet)

    def project_dist(self, next_dist, batch, gamma):
        batch_size = len(batch.states)
        vmin = self.dgnet.qnet.vmin
        vmax = self.dgnet.qnet.vmax
        natoms = self.dgnet.qnet.natoms
        delta_z = (vmax - vmin) / (natoms - 1)
        t_z = batch.rewards.view(-1, 1) + (1 - batch.terminals.view(-1, 1)) * gamma * self.dgnet.qnet.support
        t_z = t_z.clamp(min=vmin, max=vmax)
        b = (t_z - vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size * self.nagent - 1) * natoms, batch_size * self.nagent).long().unsqueeze(dim=1).expand(batch_size * self.nagent,
                                                                                                         natoms).to(
            self.device)
        proj_dist = torch.zeros_like(next_dist.view(-1, natoms), device=self.device)


        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist




