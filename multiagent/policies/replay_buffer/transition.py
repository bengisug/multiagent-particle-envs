from collections import namedtuple


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminal"))
MATransition = namedtuple("Transition", ("agent_id", "state", "action", "reward", "next_state", "terminal"))
MAGTransition = namedtuple("Transition", ("states", "actions", "next_states", "rewards", "terminals", "adjs"))
SACTransition = namedtuple("Transition", ("state", "action", "next_state", "reward", "terminal", "logprob"))