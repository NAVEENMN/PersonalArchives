import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import deque

state_space = 3
action_space = 1
rewards_space = 1

timesteps = 2
num_input = 3
num_hidden = 50
batch_size = 50
rolls_per_batch = 10
MAX_EP_STEP = 200 # maxumum number of steps per episode

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
        self.opt = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-4)

    def get_value(self, state, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):
            fc1 = tf.layers.dense(state, 100, activation=tf.nn.relu, name=n+"fc1")
            out = tf.layers.dense(fc1, 1, activation=None, name=n+"out")
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
        self.opt = tf.train.RMSPropOptimizer(learning_rate=0.001, epsilon=1e-4)

    def get_value(self, state, action, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):
            sfc1 = tf.layers.dense(state, 100, activation=tf.nn.relu, name=n+"sfc1")
            afc1 = tf.layers.dense(action, 100, activation=tf.nn.relu, name=n+"afc1")
            fc1 = tf.multiply(sfc1, afc1, name=n+"stat")
            out = tf.layers.dense(fc1, 1, activation=None, name=n+"out")
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
    def __init__(self, sess, env, name):
        self.sess = sess
        self.name = name
        self.ENTROPY_BETA = 0.01
        self.A_BOUND = [env.action_space.low, env.action_space.high]
        self.lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, epsilon=1e-4)

    # policy returns mean and sd
    def get_action(self, states, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):

            # Current data input shape: (batch_size, timesteps, n_input)
            x = tf.unstack(states, timesteps, 1, name=n+"unstack")

            outputs, states = rnn.static_rnn(self.lstm_cell, x, dtype=tf.float32)

            fc1 = tf.layers.dense(outputs[-1], 200, activation=tf.nn.relu, name=n+"fc1")
            mu = tf.layers.dense(fc1, 1, activation=tf.nn.tanh, name=n + "mu")
            sigma = tf.layers.dense(fc1, 1, activation=tf.nn.softplus, name=n+"sigma")

            # tanh gives (-1, 1) scale up in range of action bounds
            mu, sigma = mu * self.A_BOUND[1], sigma + 1e-4

        return mu, sigma

    def get_scaled_grads(self, pi, normal_dist, adv, params, gparams):
        # take log of distribution
        log_prob = normal_dist.log_prob(pi)
        exp_v = log_prob * adv
        entropy = normal_dist.entropy() # add noise to explore
        exp_v = self.ENTROPY_BETA * entropy + exp_v

        # optimzers minimize by subtracting grads we need to maximize the utility
        # U(pi) -ve of exp_v
        # 1/m * sum(grads(prob(trajectory)) * advantage)
        # if rewards are -9.0, -8.0,... -0.01 try -exp_v
        # if rewards are 0.1, 0.1, ...., -1.0 try exp_v (not sure)
        a_loss = tf.reduce_mean(-exp_v)
        # taking grads after sum
        scaled_grads = tf.gradients(a_loss, params)
        return zip(scaled_grads, gparams)

    def train_step(self, grads):
        return self.opt.apply_gradients(grads)

    def get_params(self):
        tvars = tf.trainable_variables()
        vars = [var for var in tvars if self.name in var.name]
        return vars

class game():
    def __init__(self, sess):
        self.env = gym.make('Pendulum-v0')
        self.A_BOUND = [self.env.action_space.low, self.env.action_space.high]
        self.sess = sess
        self.episodes = 0
        self.avg_rewards = []

        with tf.name_scope("Inputs"):
            self.input_state = tf.placeholder("float", [None, state_space], name="input_state")
            self.input_state_pi = tf.placeholder("float", [None, timesteps, num_input], name="input_state_pi")
            self.input_action = tf.placeholder("float", [None, action_space], name="input_action")
            self.input_adv = tf.placeholder("float", [None, rewards_space], name="input_advantage")
            self.target_vs = tf.placeholder("float", [None, rewards_space], name="target_vs")
            self.target_qv = tf.placeholder("float", [None, rewards_space], name="target_qv")
            self.target_ad = tf.placeholder("float", [None, rewards_space], name="target_ad")

        with tf.name_scope("policy"):
            pi = policy(sess, self.env, "pi_")

            self.mu, self.sigma = pi.get_action(self.input_state_pi)
            # get normal distribution for action
            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            # draw a sample from this distribution
            self.pi_u = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0),
                                         self.A_BOUND[0], self.A_BOUND[1])
            pi_vars = pi.get_params()


        with tf.name_scope("gpolicy"):
            global_pi = policy(sess, self.env, "gpi_")
            self.gmu, self.gsigma = global_pi.get_action(self.input_state_pi)
            gnormal_dist = tf.contrib.distributions.Normal(self.gmu, self.gsigma)

            global_pi_vars = global_pi.get_params()

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

        self.pi_grads = pi.get_scaled_grads(self.pi_u,
                                            gnormal_dist,
                                            self.input_adv,
                                            pi_vars, global_pi_vars)

        #pull from global

        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(pi_vars, global_pi_vars)]
        self.train_pi = global_pi.train_step(self.pi_grads)

    # (batch_size, timesteps, n_input)

    def pull_global(self):
        self.sess.run(self.pull_a_params_op)

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

    # training based on off policy ( td with monte carlo learning)
    def train_state_value(self, memory, epoch):
        st, at, rt, st1, advantage, _, done = memory.sample_batch(batch_size)
        targets = []
        v_st1 = self.get_state_value(state=st1)
        for i in range(0, batch_size):
            if done[i][0]:
                targets.append(np.reshape([0.0], [1, 1]))
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
        targets = []
        st, at, rt, st1, advantage, previous_states, done = memory.sample_batch(batch_size)
        at1 = self.get_action(ba=[st, st1]) # on policy
        q_st1 = self.get_q_value(state=st1, action=at1)
        for i in range(0, batch_size):
            if done[i][0]:
                targets.append(np.reshape([0.0], [1, 1]))
            else:
                q_target = rt[i] + 0.9 * q_st1[i]
                targets.append(np.reshape(q_target, [1, 1]))
        targets = np.reshape(targets, [-1, 1])
        feed_dict = {self.input_state: st,
                     self.input_action: at,
                     self.target_qv: targets}
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

    def compute_returns(self, states):
        buffer_v_target = []
        # last has to be terminal
        v_s_ = np.asarray([[0.0]])

        for st in states[::-1]:
            v_s_ = np.add(st.reward_gotten, np.multiply(0.9, v_s_))
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        for i in range(0, len(buffer_v_target)):
            state = states[i]
            state.reward_gotten = buffer_v_target[i]
            states[i] = state

        return states

    def run_episodes(self, epoch, roll_outs, memory):
        tr = list()
        STATES = []
        at = 0.0
        net_rewards = 0.0
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
            steps = 0
            while not done:
                steps += 1
                self.env.render()
                at = self.get_action(list(ba))
                st1, rt1, done, info = self.env.step(at)
                st1 = np.reshape(st1, [1, state_space])

                if steps > MAX_EP_STEP:
                    done = True

                # this is a confined environment
                # our objective is different from racing or cart pole env.
                # normalize rewards
                rt1 = 0.0 if done else (rt1+8.0)/8.0
                net_rewards += rt1

                # advantage is calculated wrt to current state
                q_value = self.get_q_value(state=st, action=at)
                state_value = self.get_state_value(state=st)
                advantage = q_value - rt1

                st = np.reshape(st, [1, state_space])

                stattr = state_attr(st)
                stattr.next_state = np.reshape(st1, [1, state_space])
                stattr.action_taken = np.reshape(at, [1, action_space])
                stattr.reward_gotten = np.reshape(rt1, [1, rewards_space])
                stattr.advantage = np.reshape(advantage, [1, rewards_space])
                stattr.prev_state = list(ba)
                stattr.done = True if done else False

                if done:
                    if self.episodes < 5:
                        self.avg_rewards.append(np.asarray([net_rewards]))
                    else:
                        # average for last 5 episodes
                        self.avg_rewards[-1] = (np.mean(self.avg_rewards[-5:]))

                states.append(stattr)
                ba.append(st1)
                ba.pop(0)
                st = st1

            # discount returns for one episode
            STATES.extend(self.compute_returns(list(states)))

        # batch them after m episodes
        for state in STATES:
            tr.append(state)
            memory.add(state)
        return tr

    def train(self, tr, epoch, memory):

        st = [np.reshape(v.prev_state, [timesteps, num_input]) for v in tr]
        adv = [np.reshape(v.advantage, [-1, rewards_space]) for v in tr]

        # train state value
        self.train_state_value(memory, epoch=epoch)

        # training q_value
        self.train_q_value(memory, epoch=epoch)

        # train policy, updates global vars
        self.train_policy(st=st, adv=adv, epoch=epoch)

        # pull trained global vars to local
        self.pull_global()


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
    memory = ReplayBuffer(1000000)
    sess.run(tf.global_variables_initializer())
    #for epoch in range(0, 1000):
    epoch = 0
    while True:
        epoch += 1
        gm.episodes += 1
        # run m episodes under current policy
        tr = gm.run_episodes(epoch, rolls_per_batch, memory)
        # train state_value, q_value and policy
        gm.train(tr, epoch, memory)
        print("episode {} : running rewards {}".format(gm.episodes, gm.avg_rewards[-1]))

if __name__ == "__main__":
    main()
