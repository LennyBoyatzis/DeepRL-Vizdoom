import time
import itertools
from tqdm import trange
from vizdoom import DoomGame
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
from vizdoom import Mode

class Game():
    def __init__(self, config):
        self.game = DoomGame()
        self.game.load_config(config["scenario"])
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.config = config
        self.frame_repeat = 12
        self.epochs = 20
        self.test_episodes_per_epoch = 100
        self.learning_steps_per_epoch = 2000

    def play(self, agent):
        self.game.init()
        episodes = self.config["no_of_episodes"]
        for iteration in range(episodes):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                current_state = self.get_state()
                action = agent.select_action(current_state, iteration, self.config["no_of_episodes"])
                reward = self.change_state(action)
                new_state = self.get_state()
                if new_state and reward:
                    agent.update_policy(current_state, action, new_state, reward, False)
                time.sleep(0.02)
        print("Total reward: " + str(self.game.get_total_reward()))

    def train(self, agent):
        for epoch in range(self.epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            self.game.new_episode()
            for learning_step in trange(self.learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch)
                if self.game.is_episode_finished():
                    score = self.game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(self.test_episodes_per_epoch, leave=False):
                self.game.new_episode()
                while not self.game.is_episode_finished():
                    state = preprocess(self.game.get_state().screen_buffer)
                    best_action_index = self.get_best_action(state)

                    self.game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", model_savefile)
            saver.save(session, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()

    def stop(self):
        return self.game.close()

    def get_state(self):
        return self.game.get_state()

    def change_state(self, action):
        return self.game.make_action(action, self.frame_repeat)
