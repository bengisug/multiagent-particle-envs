from multiagent.policies.model.agent import BaseAgent
from multiagent.policies.replay_buffer.uniform_buffer import UniformBuffer


class OffPolicyAgent(BaseAgent):

    def __init__(self, transition_type, buffer_size, batch_size, tau, gamma, grad_clip, device):
        super(OffPolicyAgent, self).__init__(gamma, grad_clip, device)
        self.tau = tau
        self.batch_size = batch_size
        self.transition_type = transition_type
        self.transition_buffer = UniformBuffer(buffer_size, transition_type)

    def push_transition(self, *transition):
        self.transition_buffer.push(*transition)

    def update_target(self):
        raise NotImplementedError

    def _update_target(self, net, target_net, tau=None):
        if tau is None: tau = self.tau
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data += tau * (param.data - target_param.data)


