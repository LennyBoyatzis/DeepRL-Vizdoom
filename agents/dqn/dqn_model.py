import tensorflow as tf

resolution = (30, 45)
learning_rate = 0.00025

class Model():
    def __init__(self, session, available_actions_count):
            self.state_input = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="state")
            action_input = tf.placeholder(tf.int32, [None], name="Action")
            self.target_q = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

            conv1 = tf.contrib.layers.convolution2d(self.state_input, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            conv2_flat = tf.contrib.layers.flatten(conv2)
            fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                    biases_initializer=tf.constant_initializer(0.1))

            self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.constant_initializer(0.1))

            self.best_a = tf.argmax(self.q, 1)
            self.loss = tf.losses.mean_squared_error(self.q, self.target_q)
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.train_step = optimizer.minimize(self.loss)

    def learn(self, s1, target_q):
        feed_dict = {self.state_input: s1, self.target_q: target_q}
        l, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
        return l

    def get_q_values(self, state):
        return self.session.run(self.q, feed_dict={self.state_input: state})

    def get_best_action(self, state):
        return self.session.run(self.best_a, feed_dict={self.state_input: state})

    def simple_get_best_action(self, state):
        return self.get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]
