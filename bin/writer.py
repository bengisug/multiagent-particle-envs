import yaml, os
import torch
from pathlib import Path
import networkx as nx
# import matplotlib.pyplot as plt


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


# def plot_graph(args, adj, adj_name, index):
#     G = nx.from_numpy_matrix(adj.cpu().detach().numpy(), parallel_edges=True, create_using=nx.MultiDiGraph())
#     plt.figure()
#     nx.draw_networkx(G, arrows=True)
#     plt.savefig(args.save_folder + "/plots/" + adj_name + str(index))
#     plt.close()

