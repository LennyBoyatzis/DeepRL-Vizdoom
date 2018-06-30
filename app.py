import time
import itertools
from game import Game
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
from vizdoom import Mode
from agents.random.random_agent import RandomAgent
from agents.greedy.epsilon_greedy import EpsilonGreedyAgent
from agents.dqn.dqn_agent import DQNAgent
from tensorflow.python.client import device_lib

config = {
    "scenario": "./scenarios/simpler_basic.cfg",
    "epochs": 20,
    "frame_repeat": 12,
    "test_episodes_per_epoch": 100,
    "learning_steps_per_epoch": 2000,
    "model_savefile": "./tmp/model.ckpt",
}

def init_agent(game, load_model=False):
    n_actions = game.get_available_buttons_size()
    actions = [list(a) for a in itertools.product([0, 1], repeat=n_actions)]
    return DQNAgent(actions, load_model)

def train():
    doom = Game(config)
    doom.game.load_config(config["scenario"])
    doom.game.set_window_visible(False)
    doom.game.set_mode(Mode.PLAYER)
    doom.game.set_screen_format(ScreenFormat.GRAY8)
    doom.game.set_screen_resolution(ScreenResolution.RES_640X480)

    agent = init_agent(doom.game, False)
    doom.train(agent)

def play():
    doom = Game(config)
    doom.game.set_window_visible(True)
    doom.game.set_mode(Mode.ASYNC_PLAYER)
    doom.game.set_screen_format(ScreenFormat.GRAY8)
    doom.game.set_screen_resolution(ScreenResolution.RES_640X480)

    agent = init_agent(doom.game, True)
    doom.play(agent)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == "__main__":
    # train()
    gpus = get_available_gpus()
    print("here are the gpus")
