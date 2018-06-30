import random
import tensorflow as tf
import numpy as np
from agents.dqn.dqn_replay_memory import ReplayMemory
from agents.dqn.dqn_model import Model
from agents.dqn.dqn_preprocess import preprocess, resolution

class DQNAgent():
    def __init__(self, actions = [], load_model=False):
        self.actions = actions
        self.discount_factor = 0.99
        session = tf.Session()
        self.model = Model(session, len(actions))
        saver = tf.train.Saver()
        if load_model:
            saver.restore(session, "./tmp/model.ckpt")
        else:
            init = tf.global_variables_initializer()
            session.run(init)
        self.memory = ReplayMemory()

    def get_exploration_rate(self, epoch, epochs):
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs
        eps_decay_epochs = 0.6 * epochs

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            return start_eps - (epoch - const_eps_epochs) / (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def select_action(self, game_state, iteration, epochs):
        epsilon = self.get_exploration_rate(iteration, epochs)

        if random.random() < epsilon:
            action = random.choice(self.actions) 
        else:
            state = preprocess(game_state.screen_buffer)
            best_action_index = self.model.simple_get_best_action(state)
            action = self.actions[best_action_index]
        return action

    def update_policy(self, current_state, chosen_action, new_state, reward, is_done):
        current_state = preprocess(current_state.screen_buffer)
        new_state = preprocess(new_state.screen_buffer) if not is_done else None
        chosen_action = self.actions.index(chosen_action)
        self.memory.add(current_state, chosen_action, new_state, reward, is_done)

        if self.memory.size > 64:
            s1, a, s2, isterminal, r = self.memory.get_sample(64)
            q2 = np.max(self.model.get_q_values(s2), axis=1)
            target_q = self.model.get_q_values(s1)
            target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
            self.model.learn(s1, target_q)

