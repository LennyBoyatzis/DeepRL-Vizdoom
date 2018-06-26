import random

class RandomAgent():
    def __init__(self, actions = []):
        self.actions = actions

    def select_action(self, game_state):
        return random.choice(self.actions) 

    def update_policy(self, chosen_action, reward):
        pass
