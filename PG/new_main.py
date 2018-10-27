import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import deque

rolls_per_batch = 3
state_space = 3
action_space = 1
rewards_space = 1
timesteps = 2
num_input = 3
num_hidden = 10
batch_size = 30

class state_attr():
    def __init__(self, st):
        self.prev_state = None
        self.current_state = st
        self.action_taken = None
        self.next_state = None
        self.reward_gotten = None
        self.advantage = None
        self.done = False

class v_s():
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)

    def get_value(self, state, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):
            fc1 = tf.layers.dense(state, 32, activation=tf.nn.sigmoid, name=n+"fc1")
            fc2 = tf.layers.dense(fc1, 20, activation=tf.nn.sigmoid, name=n+"fc2")
            out = tf.layers.dense(fc2, 1, activation=tf.nn.sigmoid, name=n+"out")
        return out

    def get_loss(self, source, target):
        loss = tf.losses.mean_squared_error(labels=target, predictions=source)
        return tf.reduce_mean(loss)

    def train_step(self, loss, t_vars):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_ = self.opt.minimize(loss, var_list=t_vars)
        return train_

    def get_params(self):
        tvars = tf.trainable_variables()
        vars = [var for var in tvars if self.name in var.name]
        return vars

class q_v():
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-4)

    def get_value(self, state, action, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):
            sfc1 = tf.layers.dense(state, 32, activation=tf.nn.sigmoid, name=n+"sfc1")
            afc1 = tf.layers.dense(action, 32, activation=tf.nn.sigmoid, name=n+"afc1")
            fc1 = tf.add(sfc1, afc1, name=n+"stat")
            fc2 = tf.layers.dense(fc1, 20, activation=tf.nn.sigmoid, name=n+"fc2")
            out = tf.layers.dense(fc2, 1, activation=tf.nn.sigmoid, name=n+"out")
        return out

    def get_loss(self, source, target):
        loss = tf.losses.mean_squared_error(labels=target, predictions=source)
        return tf.reduce_mean(loss)

    def train_step(self, loss, t_vars):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_ = self.opt.minimize(loss, var_list=t_vars)
        return train_

    def get_params(self):
        tvars = tf.trainable_variables()
        vars = [var for var in tvars if self.name in var.name]
        return vars

class policy():
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, name=self.name+"rnn")
        self.opt = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=1e-4)

    def get_action(self, states, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):

            # Current data input shape: (batch_size, timesteps, n_input)
            x = tf.unstack(states, timesteps, 1, name=n+"unstack")
            outputs, states = rnn.static_rnn(self.lstm_cell, x, dtype=tf.float32)

            fc1 = tf.layers.dense(outputs[-1], 10, activation=tf.nn.tanh, name=n+"fc1")
            fc2 = tf.layers.dense(fc1, 20, activation=tf.nn.tanh, name=n+"fc2")
            out = tf.layers.dense(fc2, 1, activation=tf.nn.tanh, name=n+"out")

        out = tf.multiply(out, 2.0)
        return out

    def get_scaled_grads(self, pi, adv, params):
        self.scaled_grads = tf.gradients((tf.log(pi) * adv)/rolls_per_batch, params)
        return zip(self.scaled_grads, params)

    def train_step(self, grads):
        return self.opt.apply_gradients(grads)

    def get_params(self):
        tvars = tf.trainable_variables()
        vars = [var for var in tvars if self.name in var.name]
        return vars

class game():
    def __init__(self, sess):
        self.env = gym.make('Pendulum-v0')
        self.sess = sess

        with tf.name_scope("Inputs"):
            self.input_state = tf.placeholder("float", [None, state_space], name="input_state")
            self.input_state_pi = tf.placeholder("float", [None, timesteps, num_input], name="input_state_pi")
            self.input_action = tf.placeholder("float", [None, action_space], name="input_action")
            self.input_adv = tf.placeholder("float", [None, rewards_space], name="input_advantage")
            self.target_vs = tf.placeholder("float", [None, rewards_space], name="target_vs")
            self.target_qv = tf.placeholder("float", [None, rewards_space], name="target_qv")
            self.target_ad = tf.placeholder("float", [None, rewards_space], name="target_ad")

        with tf.name_scope("policy"):
            pi = policy(sess, "pi_")
            self.pi_u = pi.get_action(self.input_state_pi)

        with tf.name_scope("value_st"):
            vs = v_s(sess, "vs_")
            self.state_value = vs.get_value(state=self.input_state)
            self.vs_loss = vs.get_loss(source=self.state_value, target=self.target_vs)
            self.train_sv = vs.train_step(loss=self.vs_loss, t_vars=vs.get_params())

        with tf.name_scope("q_value"):
            qsa = q_v(sess, "qv_")
            self.q_value = qsa.get_value(state=self.input_state,
                                         action=self.input_action)
            self.qv_loss = qsa.get_loss(source=self.q_value, target=self.target_qv)
            self.train_qv = qsa.train_step(loss=self.qv_loss, t_vars=qsa.get_params())

        self.pi_grads = pi.get_scaled_grads(self.pi_u, self.input_adv, pi.get_params())
        self.train_pi = pi.train_step(self.pi_grads)

    # (batch_size, timesteps, n_input)
    def get_action(self, ba):
        ba = np.asarray(ba)
        state = np.reshape(ba, [-1, timesteps, num_input])
        feed_dict = {self.input_state_pi: state}
        action = self.sess.run(self.pi_u, feed_dict=feed_dict)
        return action

    def get_state_value(self, state):
        state = np.reshape(state, [-1, state_space])
        feed_dict = {self.input_state: state}
        value = self.sess.run(self.state_value, feed_dict=feed_dict)
        return value

    def get_state_loss(self, state, target_vs):
        state = np.reshape(state, [-1, state_space])
        feed_dict = {self.input_state: state, self.target_vs: target_vs}
        value = self.sess.run(self.vs_loss, feed_dict=feed_dict)
        return value

    def get_q_value(self, state, action):
        state = np.reshape(state, [-1, state_space])
        action = np.reshape(action, [-1, rewards_space])
        feed_dict = {self.input_state: state, self.input_action: action}
        value = self.sess.run(self.q_value, feed_dict=feed_dict)
        return value

    def get_q_value_loss(self, state, action):
        state = np.reshape(state, [-1, state_space])
        action = np.reshape(action, [-1, rewards_space])
        feed_dict = {self.input_state: state, self.input_action: action}
        value = self.sess.run(self.qv_loss, feed_dict=feed_dict)
        return value

    def train_state_value(self, memory, epoch):
        st, at, rt, st1, advantage, _, done = memory.sample_batch(batch_size)
        targets = []
        v_st1 = self.get_state_value(state=st1)
        for i in range(0, batch_size):
            if done[i][0]:
                targets.append(np.reshape(rt[i], [1, 1]))
            else:
                v_target = rt[i] + 0.9 * v_st1[i]
                targets.append(np.reshape(v_target, [1, 1]))
        targets = np.reshape(targets, [-1, 1])
        feed_dict = {self.input_state: st,
                     self.target_vs: targets}
        self.sess.run(self.train_sv, feed_dict=feed_dict)
        if epoch%10 == 0:
            print("vs_loss", self.sess.run(self.vs_loss, feed_dict=feed_dict))

    def train_q_value(self, memory, epoch):
        st, at, rt, st1, advantage, previous_states, done = memory.sample_batch(batch_size)
        at1 = self.get_action(ba=[st, st1]) # on policy
        q_st1 = self.get_q_value(state=st1, action=at1)
        q_target = rt + 0.9 * q_st1
        feed_dict = {self.input_state: st,
                     self.input_action: at,
                     self.target_qv: q_target}
        self.sess.run(self.train_qv, feed_dict=feed_dict)
        if epoch % 10 == 0:
            print("qv_loss", self.sess.run(self.qv_loss, feed_dict=feed_dict))

    def train_policy(self, st, adv, epoch):
        adv = np.reshape(adv, [-1, rewards_space])
        feed_dict = {self.input_state_pi: st,
                     self.input_adv: adv}
        self.sess.run(self.train_pi, feed_dict=feed_dict)
        if epoch % 10 == 0:
            print("adv", max(np.reshape(adv, [-1])))

    def compute_returns(self, tr):
        returns = [-1.0]
        for _ in tr[::-1]:
            value = 0.1 + 0.9 * returns[-1]
            returns.append(value)
        returns = returns[::-1]
        returns = returns[1:]
        for i in range(0, len(tr)):
            state = tr[i]
            state.reward_gotten = returns[i]
            tr[i] = state
        return tr

    def run_episodes(self, epoch, roll_outs, memory):
        tr = list()
        states = []
        at = 0.0
        for roll in range(0, roll_outs):
            states = list()
            done = False
            self.env.reset()
            action = self.env.action_space.sample()
            st0, _, _, _ = self.env.step(action)
            st0 = np.asarray(st0)
            st, _, _, _ = self.env.step(action)
            st0 = np.reshape(np.asarray(st0), [1, state_space])
            st = np.reshape(np.asarray(st), [1, state_space])
            ba = [st0, st]
            while not done:
                self.env.render()
                at = self.get_action(list(ba))
                st1, rt1, done, info = self.env.step(at)
                st1 = np.asarray(st1)

                rt1 = 0.1
                if done:
                    rt1 = -1.0

                # advantage is calculated wrt to current state
                q_value = self.get_q_value(state=st, action=at)
                state_value = self.get_state_value(state=st)
                advantage = q_value - state_value

                st = np.reshape(st, [1, state_space])
                st1 = np.reshape(st1, [1, state_space])
                at = np.reshape(at, [1, action_space])
                rt1 = np.reshape(rt1, [1, rewards_space])
                advantage = np.reshape(advantage, [1, rewards_space])

                stattr = state_attr(st)
                stattr.action_taken = at
                stattr.next_state = st1
                stattr.prev_state = list(ba)
                stattr.reward_gotten = rt1
                stattr.advantage = advantage
                if done:
                    stattr.done = True
                states.append(stattr)
                ba.append(st1)
                ba.pop(0)
                st = st1

        states = self.compute_returns(states)
        for state in states:
            tr.append(state)
            memory.add(state)

        print("epoch {} : at{} total return {}".format(epoch, at, len(tr)))
        return tr

    def train(self, tr, epoch, memory):

        st = [np.reshape(v.prev_state, [timesteps, num_input]) for v in tr]
        adv = [np.reshape(v.advantage, [-1, rewards_space]) for v in tr]

        self.train_state_value(memory, epoch=epoch)

        # training q_value
        self.train_q_value(memory, epoch=epoch)

        # train policy
        self.train_policy(st=st, adv=adv, epoch=epoch)


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, tr):
        '''
        st = tr.current_state
        at = tr.action_taken
        rt1 = tr.reward_gotten
        st1 = tr.next_state
        advantage = tr.advantage
        '''
        experience = tr
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.count -= 1
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        st = np.reshape([_.current_state for _ in batch], [-1, state_space])
        at = np.reshape([_.action_taken for _ in batch], [-1, action_space])
        rt1 = np.reshape([_.reward_gotten for _ in batch], [-1, rewards_space])
        st1 = np.reshape([_.next_state for _ in batch], [-1, state_space])
        advantage = np.reshape([_.advantage for _ in batch], [-1, rewards_space])
        previous_states = np.reshape([_.prev_state for _ in batch], [-1, timesteps, num_input])
        done = np.reshape([_.done for _ in batch], [-1, 1])

        return st, at, rt1, st1, advantage, previous_states, done

def main():
    sess = tf.InteractiveSession()
    gm = game(sess)
    memory = ReplayBuffer(10000)
    sess.run(tf.global_variables_initializer())
    for epoch in range(0, 1000):
        # run m episodes under current policy
        tr = gm.run_episodes(epoch, rolls_per_batch, memory)
        # train state_value, q_value and policy
        gm.train(tr, epoch, memory)

if __name__ == "__main__":
    main()
