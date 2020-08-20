import multiagent.scenarios.simple_graph_comm as mdp
from multiagent.environment import MultiAgentEnv
from multiagent.policies.model.dgn_agent3 import DGNAgent
from multiagent.policies.model.adj_agent import ADJAgent
from multiagent.policies.network.graph_qnet3 import GraphQNet1
from multiagent.policies.network.adj_net import ADJActor, ADJCritic
# from multiagent.policies.network.graph_qnet2 import GraphQNet1
from multiagent.policies.replay_buffer.transition import MAGTransition, SACTransition
from bin import writer
from tensorboardX import SummaryWriter
import argparse
import importlib
import numpy as np
import os
import torch
from copy import deepcopy
from datetime import datetime as dt


def get_parser():
    time_sign = str(dt.now()).replace(' ', '').replace(':', '')
    parser = argparse.ArgumentParser(
        description="Reimplementation of Graph Convolutional Reinforcement Learning!")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate of q-network (default: %(default)s)")
    parser.add_argument("--lradj", type=float, default=0.003,
                        help="Learning rate of q-network (default: %(default)s)")
    parser.add_argument("--gamma", type=float, default=0.96,
                        help="Discount rate (default: %(default)s)")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Target update rate (default: %(default)s)")
    parser.add_argument("--clip-grad", action="store_true",
                        help=("Clip gradients if True"
                              " (default: %(default)s)"))
    parser.add_argument("--reg-coef", type=float, default=0.03,
                        help=("Regularization coefficient"
                              "(default: %(default)s)"))
    parser.add_argument("--start-epsilon", type=float, default=0.5,
                        help=("Initial epsilon value "
                              "(default: %(default)s)"))
    parser.add_argument("--end-epsilon", type=float, default=0.01,
                        help=("Terminal epsilon value "
                              "(default: %(default)s)"))
    parser.add_argument("--epsilon-decay", type=float, default=0.96,
                        help=("Epsilon decay rate"
                              "(default: %(default)s)"))
    parser.add_argument("--nagent", type=int, default=10,
                        help="Number of agents (default: %(default)s)")
    parser.add_argument("--nopponent", type=int, default=12,
                        help="Number of opponents (default: %(default)s)")
    parser.add_argument("--map-size", type=int, default=30,
                        help="Number of agents (default: %(default)s)")
    parser.add_argument("--episodes", type=int, default=100000,
                        help="Number of episodes (default: %(default)s)")
    parser.add_argument("--pretrain-episodes", type=int, default=2,
                        help="Number of episodes before training starts (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=300,
                        help="Maximum time step of the environment (default: %(default)s)")
    parser.add_argument("--buffer-size", type=int, default=200000,
                        help="Replay buffer size (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size (default: %(default)s)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Pytorch device (default: %(default)s)",
                        choices=["cuda", "cpu"])
    parser.add_argument("--file-name", type=str, default="agent",
                        help="File name (default: %(default)s)")
    parser.add_argument("--agent-type", type=str, default="A2CDGNAgent",
                        help="Agent type (default: %(default)s)",
                        choices=["DQNAgent", "DGNAgent", "A2CDGNAgent"])
    parser.add_argument("--opponent-type", type=str, default="DQNAgent",
                        help="Opponent type (default: %(default)s)",
                        choices=["DQNAgent", "DGNAgent"])
    parser.add_argument("--environment-type", type=str, default="multi-agent",
                        help="Environment type (default: %(default)s)",
                        choices=["single-agent", "multi-agent"])
    parser.add_argument("--environment-name", type=str, default="BattleEnv",
                        help="Environment name (default: %(default)s)",
                        choices=["BattleEnv", "JungleEnv"])
    parser.add_argument("--gym-environment-name", type=str, default="LunarLander-v2",
                        help="Environment name (default: %(default)s)",
                        choices=["LunarLander-v2"])
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model (default: False)")
    parser.add_argument("--eval-period", type=int, default=50,
                        help=("Evaluate the model every nth episode "
                              "(default: %(default)s)"))
    parser.add_argument("--save-folder", type=str,
                        default='saved_results/' + time_sign,
                        help="Folder name for experiment results and logs (default: %(default)s)")
    parser.add_argument("--plot", action="store_true",
                        help=("Plot episode rewards and losses if True"
                              " (default: %(default)s)"))
    parser.add_argument("--plot-grad", action="store_true",
                        help=("Plot gradients if True"
                              " (default: %(default)s)"))
    parser.add_argument("--plot-network", action="store_true",
                        help=("Plot network if True"
                              " (default: %(default)s)"))
    parser.add_argument("--writer-name", type=str, default='saved_results/' + time_sign + '/plots',
                        help="Summary writer name (default: %(default)s)")
    parser.add_argument("--hidden-layer-size", type=int, default=1,
                        help=("Hidden layer size for networks "
                              "(default: %(default)s)"))
    parser.add_argument("--node-size", type=int, default=256,
                        help=("Hidden node size for networks "
                              "(default: %(default)s)"))
    parser.add_argument("--vmin", type=float, default=-10,
                        help="Minimum value for distributional DQN extension")
    parser.add_argument("--vmax", type=float, default=10,
                        help="Maximum value for distributional DQN extension")
    parser.add_argument("--natoms", type=int, default=51,
                        help="Number of atoms in distributional DQN extension")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_parser()
    os.mkdir(args.save_folder)
    writer.save_args(args)
    summary_writer = SummaryWriter(args.writer_name)

    scenario = mdp.Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=False)

    gatqnet = GraphQNet1(3, 5, args)

    agent = DGNAgent(5, args.nagent, gatqnet, args.lr, args.reg_coef, args.start_epsilon,
             args.end_epsilon, args.epsilon_decay, MAGTransition, args.buffer_size,
             args.batch_size, args.tau, args.gamma, args.clip_grad, args.device, 0)

    adj_actor = ADJActor(30, 128, args.nagent)
    adj_critic1 = ADJCritic(30, 128, args.nagent)
    adj_critic2 = ADJCritic(30, 128, args.nagent)

    adj_agent = ADJAgent(adj_actor, adj_critic1, adj_critic2, args.lradj, args.lradj, args.gamma, args.clip_grad,
                     SACTransition, args.buffer_size, args.batch_size, args.tau,
                     args.nagent, args.device)

    best_score = -np.inf
    epsilon = args.start_epsilon

    world = env.world

    g_i = 0

    for eps in range(args.episodes):
        epsilon = max(args.end_epsilon, epsilon * args.epsilon_decay)

        state_n = env.reset()

        episode_score = 0

        for t in range(args.max_length):

            _adj = env.world.adj
            for j in range(len(world.agents)):
                for i, a in enumerate(world.agents):
                    if state_n[i][1] == 1:
                        _adj[j][i] = 1
                    else:
                        _adj[j][i] = 0
            env.world.adj = torch.from_numpy(np.array(_adj))

            for i, a in enumerate(env.agents):
                a.last_comm = state_n[i][2]
            torch_state_n = agent._totorch(state_n)

            # torch_state_z = adj_agent.encode_state(torch_state_n)

            adj_state_n = torch_state_n.view(1, 30).repeat(10, 1)

            adj, adj_logprob = adj_agent.act(adj_state_n)

            # if t % 2 != 0:
            #     adj, adj_logprob = adj_agent.act(torch_state_z)
            # else:
            #     adj = torch.ones_like(env.world.adj) - 1

            # adj, adj_logprob = adj_agent.act(torch_state_z)

            # if t == args.max_length - 1:
            #     print("ADJ @@@@@@@@@@@")
            #     print(env.world.adj == check_adj)
            #     print("ADJ $$$$$$$$$$$")
            #     print(((env.world.adj == check_adj).int().sum()))
                # print("ADJ_LP @@@@@@@@@@@")
                # print(adj)
                # print("ADJ_LP $$$$$$$$$$$")

            # adj = torch.ones_like(env.world.adj) -1
            # adj = (env.world.adj) - 1

            check_adj = deepcopy(adj)
            check_adj[check_adj >= 0] = 1
            check_adj[check_adj < 0] = 0

            adj_reward_n = (env.world.adj == check_adj).int().sum(axis=-1).true_divide(10)

            action_n, causal_influence, adj_lp = agent.act(torch_state_n, adj, epsilon)
                # print("OBS @@@@@@@@@@@")
                # print(state_n)
                # print("OBS $$$$$$$$$$$")
                # print("ACTION @@@@@@@@@@@")
                # print(action_n)
                # print("ACTION $$$$$$$$$$$")

            if args.plot and causal_influence is not np.nan:
                summary_writer.add_scalar('data/causal-influence', causal_influence, t + eps * args.max_length)

            next_state_n, reward_n, done_n, _ = env.step(action_n)

            if t == args.max_length - 1:
                print(adj_reward_n)
                print(env.world.adj)
                print(check_adj)
                print(((env.world.adj == check_adj).int().sum()))
                # writer.plot_graph1(args, env.world.adj, "simple-graph", str(g_i) + '_0', state_n, action_n, reward_n)
                # writer.plot_graph2(args, env.world.adj, "simple-graph", str(g_i) + '_1', state_n, action_n, reward_n)
                g_i += 1

            torch_next_state_n = agent._totorch(next_state_n)
            # torch_next_state_z = adj_agent.encode_state(torch_next_state_n)
            adj_next_state_n = torch_next_state_n.view(1, 30).repeat(10, 1)

            episode_score += sum(reward_n)

            # if t == args.max_length - 1:
            #     print("WTF: {}".format(adj))
            #     print("Check WTF: {}".format(check_adj))

            transition = (state_n, action_n, next_state_n, reward_n, done_n, adj)
            agent.push_transition(*transition)

            # adj_reward_n = agent.intrinsic_reward(torch_state_z, adj)

            if adj_lp is not np.nan:
                adj_transition = (adj_state_n, adj, adj_next_state_n, adj_reward_n, done_n, adj_logprob)
                adj_agent.push_transition(*adj_transition)

            if eps > args.pretrain_episodes:
                value_loss = agent.update()
                # if t % 5 == 0:
                value_loss2 = adj_agent.update(agent.intrinsic_reward)
                if args.plot:
                    summary_writer.add_scalar('data/value-loss', value_loss, t + eps * args.max_length)
                if args.plot_grad:
                    writer.plot_grad(summary_writer, agent.dgnet, "dg-net", t + eps * args.max_length)

            for a in world.agents:
                a.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

            state_n = next_state_n

        print(
            "Progress: {:.2}%, episode: {}/{}, episode score: {}, average episode score: {:.5}".format(
                eps / args.episodes * 100, eps, args.episodes, episode_score,
                episode_score / args.max_length))

        if (eps + 1) % args.eval_period == 0:
            agent.save_model(args.save_folder + "/" + args.file_name)

            if args.plot:
                summary_writer.add_scalar('data/training-score', episode_score, eps)

    agent.save_model(args.save_folder + "/" + args.file_name)
    summary_writer.export_scalars_to_json("./all_scalars.json")
    summary_writer.close()
