import random

def ind_max(x):
    m = max(x)
    return x.index(m)

class EpsilonGreedyAgent():
    def __init__(self, actions = []):
        self.actions = get_actions()
        self.epsilon = 0.2 
        self.counts = [0] * len(self.actions)
        self.values = [0.0] * len(self.actions)

    def select_action(self, game_state):
        if random.random() > self.epsilon:
            exploit_action_idx = ind_max(self.values)
            exploit_action = self.actions[exploit_action_idx]
            return self.actions[exploit_action_idx]
        else:
            explore_action_idx = random.randrange(len(self.values))
            return self.actions[explore_action_idx]

    def update_policy(self, chosen_action, reward):
        chosen_action_idx = self.actions.index(chosen_action) 
        self.counts[chosen_action_idx] = self.counts[chosen_action_idx] + 1
        n = self.counts[chosen_action_idx]

        value = self.values[chosen_action_idx]
        new_value = ((n -1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_action_idx] = new_value
        return
