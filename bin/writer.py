import yaml, os
import torch
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


def save_args(args):
    with open(args.save_folder + '/args.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def evaluate_and_save(agent, env, args, best_reward, evaluate, opponent=None):
    if opponent is None:
        eval_reward = evaluate(args, agent=agent, env=env)
    else:
        eval_reward = evaluate(args, agent=agent, opponent=opponent, env=env)
    if eval_reward > best_reward:
        old_file = Path(args.save_folder + "/best_" + args.file_name + "_{:.3f}.p".format(best_reward))
        if old_file.exists():
            os.remove(old_file)
        best_reward = eval_reward
        agent.save_model(args.save_folder + "/best_" + args.file_name + "_{:.3f}".format(best_reward))
    return best_reward


def plot_grad(summary_writer, net, net_name, index):
    for i, param in enumerate(net.parameters()):
        name = "data/" + net_name + str(i) + "-absolute-grad"
        summary_writer.add_histogram(name, torch.abs(param.grad.detach()), index)


def plot_graph1(args, adj, adj_name, index, state, action, reward):
    G = nx.from_numpy_matrix(adj.cpu().detach().numpy(), parallel_edges=True, create_using=nx.MultiDiGraph())

    plt.figure()
    d = {}
    l = {}
    c = []
    a = 0.1
    for i in range(len(state)):
        d[i] = (state[i][1], state[i][2])
        # l[i] = (i, state[i][1])
        l[i] = ('V: ' + str(state[i][3]), 'A: ' + str(action[i][0]), 'R: ' + str(reward[i]))
        c.append((0.8, 0.4, a))
        a += 0.1
    nx.draw_networkx(G, pos=d, arrows=True, node_color=c, with_labels=True, labels=l, font_size=6)
    # nx.draw_networkx_labels(G, pos=d, labels=l)
    plt.savefig(args.save_folder + "/plots/" + adj_name + str(index))
    plt.close()

def plot_graph2(args, adj, adj_name, index, state, action, reward):
    G = nx.from_numpy_matrix(adj.cpu().detach().numpy(), parallel_edges=True, create_using=nx.MultiDiGraph())

    plt.figure()
    d = {}
    l = {}
    c = []
    a = 0
    for i in range(len(state)):
        d[i] = (state[i][1], state[i][2])
        # l[i] = (i, state[i][1])
        l[i] = i
        c.append((0.8, 0.4, a))
        a += 0.1
    nx.draw_networkx(G, pos=d, arrows=True, node_color=c, with_labels=True, label=l, font_size=12)
    # nx.draw_networkx_labels(G, pos=d, labels=l)
    plt.savefig(args.save_folder + "/plots/" + adj_name + str(index))
    plt.close()

