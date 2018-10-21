# code from https://zhuanlan.zhihu.com/p/21477488
# train AttackTarget with simple DQN


import tensorflow as tf
import gym
import numpy as np
import random
from collections import deque
import random


INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10000
GAMMA = 0.9
BATCH_SIZE = 32
HIDDEN_UNITS = 20
LEARNING_RATE = 0.01


class DQN():
    """
    Description:
        build a simple Mullti-layer perception network to recurrent DQN
        the nework has two layers and each layer is full connected. 
    """

    def __init__(self, env):
        self.action_dim = env.action_space.n

        self.state_dim = env.observation_space.shape[0]
        self.replay_buffer = deque()
        self.epsilon = INITIAL_EPSILON
        self.time_step = 0

        self.create_Q_network()
        self.create_train_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        hidden_layer = tf.layers.dense(
            self.state_input, HIDDEN_UNITS, activation=tf.nn.relu)
        output_layer = tf.layers.dense(hidden_layer, self.action_dim)
        self.Q_value = output_layer  # create Q value with One-hot represent

    def create_train_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        # reshape one-hot to single value
        Q_action = tf.reduce_sum(tf.multiply(
            self.action_input, self.Q_value), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(
            LEARNING_RATE).minimize(self.cost)

    def action(self, state):
        """ 
        Random select action from the trained network
        """
        Q_value = self.Q_value.eval(
            feed_dict={self.state_input: [state]}
        )[0]
        return np.argmax(Q_value)

    def egreedy_action(self, state):
        """
        Using egreedy to select an anction
        """
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def perceive(self, state, action, reward, next_state, done):
        """
        recieve information from environment and train the network
        """
        # Reshape ac tion to one-hot represent
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append(
            (state, one_hot_action, reward, next_state, done)
        )
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        # if there is enough samples, then start to train the netowrk
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        """
        Optimize the network
        """
        # Step1: obtain random minibatch from replay memory
        mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [batch[0] for batch in mini_batch]
        action_batch = [batch[1] for batch in mini_batch]
        reward_batch = [batch[2] for batch in mini_batch]
        next_state_batch = [batch[3] for batch in mini_batch]

        # Calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(
            feed_dict={self.state_input: next_state_batch}
        )
        for i in range(BATCH_SIZE):
            done = mini_batch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.argmax(Q_value_batch))
        self.optimizer.run(
            feed_dict={
                self.state_input: state_batch,
                self.action_input: action_batch,
                self.y_input: y_batch
            }
        )


from AttackTarget import AttackTargetEnv

if __name__ == "__main__":
    env = AttackTargetEnv()
    dqn = DQN(env)

