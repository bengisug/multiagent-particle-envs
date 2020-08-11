"""
Scenario:
n_a agents, n_g groups given as adjacency matrix, selective cooperation
Agents are rewarded for selecting the same actions as their group
"""


import numpy as np
import random
import torch
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario


class GraphCommAgent(Agent):

    def __init__(self):
        super(GraphCommAgent, self).__init__()
        self.last_comm = None


class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        world.num_agents = 10
        world.dim_c = 1
        world.adj = torch.randint(low=0, high=2, size=(world.num_agents, world.num_agents))
        mask = torch.eye(world.num_agents, world.num_agents).bool()
        world.adj.masked_fill_(mask, 1)

        world.nact = 3

        world.agents = [GraphCommAgent() for _ in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.speaker = True
            agent.movable = False

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # world.adj = torch.randint(low=0, high=2, size=(world.num_agents, world.num_agents))
        # mask = torch.eye(world.num_agents, world.num_agents).bool()
        # world.adj.masked_fill_(mask, 1)

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            agent.key = None

        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent):
        return agent.state.c

    # def reward(self, agent, world):
    #     adj = world.adj
    #     act = agent.state.c[0]
    #
    #     if agent.last_comm[0] == 0:
    #         reward = 1 if act == 2 else -1
    #     elif agent.last_comm[0] == 1:
    #         reward = 1 if act == 0 else -1
    #     else:
    #         reward = 1 if act == 1 else -1
    #
    #     return reward

    # def reward(self, agent, world):
    #     adj = world.adj
    #     act = agent.state.c[0]
    #
    #     votes = 0
    #     for a in world.agents:
    #         if adj[agent.id][a.id] == 1:
    #             votes += a.last_comm
    #     if votes <= 3:
    #         reward = 1 if act == 2 else -1
    #     elif votes <= 7:
    #         reward = 1 if act == 1 else -1
    #     else:
    #         reward = 1 if act == 0 else -1
    #     return reward

    def reward(self, agent, world):
        adj = world.adj
        act = agent.state.c[0]

        votes = np.zeros(world.nact)
        for a in world.agents:
            if adj[agent.id][a.id] == 1:
                votes[a.last_comm] += 1
        vote = np.argmax(votes)
        reward = 1 if act == vote else -1

        return reward


    def observation(self, agent, world):
        return [agent.id, random.randint(0, 2)]