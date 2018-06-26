def train_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    next_Q_values[is_terminal] = 0
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    model.fit(
        [start_states, actions], actions * Q_values[:, None],
        nb_epoch=1, batch_size=len(start_states), verbose=0
    )
