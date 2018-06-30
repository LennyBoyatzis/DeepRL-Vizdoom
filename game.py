import time
import itertools
import numpy as np
from tqdm import trange
from vizdoom import DoomGame

class Game():
    def __init__(self, config):
        self.game = DoomGame()
        self.game.load_config(config["scenario"])
        self.config = config

    def act_and_observe(self, agent, iteration):
        current_state = self.game.get_state()
        action = agent.select_action(current_state, iteration, self.config["epochs"])
        reward = self.game.make_action(action, self.config["frame_repeat"])
        new_state = self.game.get_state()
        return current_state, action, new_state, reward

    def play(self, agent):
        self.game.init()
        episodes = self.config["epochs"]
        for iteration in range(episodes):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                current_state, action, new_state, reward = self.act_and_observe(agent, iteration)
                if reward and new_state:
                    agent.update_policy(current_state, action, new_state, reward, False)
        print("Total reward: " + str(self.game.get_total_reward()))

    def train(self, agent):
        self.game.init()
        for epoch in range(self.config["epochs"]):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            self.game.new_episode()
            for learning_step in trange(self.config["learning_steps_per_epoch"], leave=False):
                while not self.game.is_episode_finished():
                    current_state, action, new_state, reward = self.act_and_observe(agent, epoch)
                    if reward and new_state:
                        agent.update_policy(current_state, action, new_state, reward, False)
                if self.game.is_episode_finished():
                    score = self.game.get_total_reward()
                    train_scores.append(score)
                    self.game.new_episode()
                    train_episodes_finished += 1
            train_scores = np.array(train_scores)
            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            agent.saver.save(session, self.config["model_savefile"])
        self.game.close()
