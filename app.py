import time
import itertools
from game import Game
from vizdoom import Button
from agents.random.random_agent import RandomAgent
from agents.greedy.epsilon_greedy import EpsilonGreedyAgent
from agents.dqn.dqn_agent import DQNAgent

config = {
    "scenario": "./scenarios/basic.cfg",
    "no_of_episodes": 1,
    "set_episode_timeout": 100,
    "set_episode_start_time": 10,
    "set_window_visible": True,
    "set_living_reward": -1,
}

def init_agent(game):
    n_actions = game.get_available_buttons_size()
    actions = [list(a) for a in itertools.product([0, 1], repeat=n_actions)]
    return DQNAgent(actions)

def start():
    doom = Game(config)
    agent = init_agent(doom.game)
    doom.train(agent)

if __name__ == "__main__":
    start()
