import gym
import numpy as np
import tensorflow as tf

rolls_per_batch = 5

class policy():
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)
        with tf.name_scope("Inputs"):
            self.input_state = tf.placeholder("float", [None, 4], name="input_state")
            self.input_adv = tf.placeholder("float", [None, 1], name="input_adv")

        with tf.name_scope("Layers"):
            fc1 = tf.layers.dense(self.input_state, 32, activation=tf.nn.relu, name="E_fc1")
            fc2 = tf.layers.dense(fc1, 20, activation=tf.nn.tanh, name="E_fc2")
            self.out = tf.layers.dense(fc2, 1, activation=tf.nn.sigmoid, name="E_out")
            params = tf.trainable_variables()
            self.grads = tf.gradients(tf.log(self.out), params)
            self.scaled_grads = tf.gradients((tf.log(self.out)*self.input_adv)/rolls_per_batch, params)
            self.train = self.opt.apply_gradients(zip(self.scaled_grads, params))

        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        state = np.reshape(state, [-1, 4])
        feed_dict = {self.input_state: state}
        action = self.sess.run(self.out, feed_dict=feed_dict)
        grads = self.sess.run(self.grads, feed_dict=feed_dict)
        return action, grads

    def get_scaled_grads(self, state, reward):
        state = np.reshape(state, [-1, 4])
        reward = np.reshape(reward, [-1, 1])
        feed_dict = {self.input_state: state, self.input_adv: reward}
        grads = self.sess.run(self.scaled_grads, feed_dict=feed_dict)
        self.sess.run(self.train, feed_dict=feed_dict)
        return grads

class game():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.policy = policy()

    def compute_returns(self, tr):
        returns = [-1.0]
        for sa_pair in tr[::-1]:
            value = 0.1 + 0.9 * returns[-1]
            returns.append(value)
        returns = returns[::-1]
        returns = returns[1:]
        #print(len(returns), returns)
        states = [sa[0] for sa in tr]
        return states, returns
        '''
        states = np.reshape(states, [-1, 4])
        returns = np.reshape(returns, [-1, 1])
        #at, gd = self.policy.get_action(states)
        print(at)
        print(gd)
        print("---")
        #
        print("--")
        for grad in grads:
            print(grad)
        print("-----")
        print(" ")
        '''


    def run_an_episodes(self, epoch, roll_outs):
        STATES, ADVS = [], []
        for roll in range(0, roll_outs):
            done = False
            self.env.reset()
            action = self.env.action_space.sample()
            st, _, _, _ = self.env.step(action)
            st = np.asarray(st)
            tr = list()
            while not done:
                #self.env.render()
                at, grads = self.policy.get_action(st)
                at = 1 if at[0][0] > 0.5 else 0
                st1, rt1, done, info = self.env.step(at)
                st1 = np.asarray(st1)
                tr.append([st, at, rt1, st1, grads])
                st = st1
            states, returns = self.compute_returns(tr)
            STATES.extend(states)
            ADVS.extend(returns)
        print("epoch {} : total return {}".format(epoch, len(STATES)))
        return STATES, ADVS

    def train(self, states, returns):
        states = np.reshape(states, [-1, 4])
        returns = np.reshape(returns, [-1, 1])
        grads = self.policy.get_scaled_grads(states, returns)
        #print(grads)

def main():
    gm = game()
    for epoch in range(0, 1000):
        states, returns = gm.run_an_episodes(epoch, rolls_per_batch)
        gm.train(states, returns)

if __name__ == "__main__":
    main()
