import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

state_space = 3
action_space = 1
rewards_space = 1

timesteps = 2
num_input = 3
num_hidden = 50
batch_size = 50
rolls_per_batch = 10
MAX_EP_STEP = 300 # maximum number of steps per episode
LOG = "logdir/"
MODEL_PATH = "saved_model/"

class EnvProp:
    def __init__(self, env):
        self.env = env
        self.a_bound = [env.action_space.low, env.action_space.high]
        self.state_space = 3
        self.action_space = 1
        self.rewards_space = 1
        # number of episodes to collect per policy train
        self.max_ep_per_train = 10
        # terminate game of number of steps exceed
        self.max_ep_step = 200
        # max episodes to run overall
        self.max_episodes = 2000

class v_s():
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
<<<<<<< HEAD

=======
>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d

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
        return [var for var in tf.trainable_variables() if self.name in var.name]
<<<<<<< HEAD


=======


>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d
class Policy:
    def __init__(self, sess, env, name):
        self.sess = sess
        self.name = name
        self.ENTROPY_BETA = 0.01
        self.A_BOUND = [env.action_space.low, env.action_space.high]
        self.opt = tf.train.RMSPropOptimizer(learning_rate=0.0001)

    # policy returns a distribution N(mu, sd)
    def get_action(self, states, reuse=False):
        n = self.name
        with tf.variable_scope(self.name, reuse=reuse):
            '''
            # Current data input shape: (batch_size, timesteps, n_input)
            x = tf.unstack(states, timesteps, 1, name=n+"unstack")
            outputs, states = rnn.static_rnn(self.lstm_cell, x, dtype=tf.float32)
            outputs[-1]
            '''
            fc1 = tf.layers.dense(states, 300, activation=tf.nn.relu, name=n+"fc1")
            mu = tf.layers.dense(fc1, 1, activation=tf.nn.tanh, name=n + "mu")
            sigma = tf.layers.dense(fc1, 1, activation=tf.nn.softplus, name=n+"sigma")

            # tanh gives (-1, 1) scale up in range of action bounds
            mu, sigma = mu * self.A_BOUND[1], sigma + 1e-4

            # get normal distribution for action
            normal_dist = tf.contrib.distributions.Normal(mu, sigma, name=n+"dist")

        return normal_dist

    def get_scaled_grads(self, pi, normal_dist, adv, params):
        # take log of distribution
        log_prob = normal_dist.log_prob(pi)
        exp_v = log_prob * adv
        entropy = normal_dist.entropy() # add noise to explore
        exp_v = self.ENTROPY_BETA * entropy + exp_v
        a_loss = tf.reduce_mean(-exp_v)

        # taking grads after sum
        scaled_grads = tf.gradients(a_loss, params)
        return scaled_grads

    def train_step(self, grads):
        return self.opt.apply_gradients(grads)

    def get_params(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

class game():
    def __init__(self, sess):
        self.game_env = EnvProp(gym.make('Pendulum-v0'))
        self.sess = sess
        self.memory = ReplayBuffer(100000)
        self.episodes = 0
        self.avg_rewards = []
        data = dict()

        with tf.name_scope("Inputs"):
            self.input_state = tf.placeholder("float", [None, state_space], name="input_state")
            self.input_action = tf.placeholder("float", [None, action_space], name="input_action")
            self.target_vs = tf.placeholder("float", [None, rewards_space], name="target_vs")

        with tf.name_scope("value_st"):
            vs = v_s(sess, "vs_")
            self.state_value = vs.get_value(state=self.input_state)
            self.vs_loss = vs.get_loss(source=self.state_value, target=self.target_vs)
            data["value_loss"] = self.vs_loss
            self.train_sv = vs.train_step(loss=self.vs_loss, t_vars=vs.get_params())

        with tf.name_scope("local_policy"):
            pi = Policy(sess, self.game_env.env, "pi_")
            normal_dist = pi.get_action(self.input_state)

            # draw a sample from this distribution
            self.pi_u = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0),
                                         self.game_env.a_bound[0],
                                         self.game_env.a_bound[1])

            self.advantage = self.target_vs - self.state_value

            self.pi_grads = pi.get_scaled_grads(self.input_action,
                                                normal_dist,
                                                self.advantage,
                                                pi.get_params())
            self.pi_vars = pi.get_params()

        with tf.name_scope("gpolicy"):
            global_pi = Policy(sess, self.game_env.env, "gpi_")
            gnormal_dist = global_pi.get_action(self.input_state)
            self.global_pi_vars = global_pi.get_params()

        #pull from global
        self.pull_a_params_op = [tf.assign(l_p, g_p) for l_p, g_p in zip(self.pi_vars, self.global_pi_vars)]
        self.train_pi = global_pi.train_step(zip(self.pi_grads, self.global_pi_vars))
<<<<<<< HEAD

        self.tensorboard_summary(data)
        with tf.name_scope("Tensorboard"):
            self.merged = tf.summary.merge_all()

=======

        self.tensorboard_summary(data)
        with tf.name_scope("Tensorboard"):
            self.merged = tf.summary.merge_all()

>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d
    def tensorboard_summary(self, data):
        with tf.name_scope("summary/losses"):
            tf.summary.scalar("value_loss", data["value_loss"])

    # (batch_size, timesteps, n_input)
    def print_local_vars(self):
        print("-- local vars --")
        print(self.pi_vars[0], self.sess.run(self.pi_vars)[0])
        print("--")

    def print_global_vars(self):
        print("-- global vars --")
        print(self.global_pi_vars[0], self.sess.run(self.global_pi_vars)[0])
        print("--")

    def pull_global(self):
        self.sess.run(self.pull_a_params_op)

    def get_action(self, state):
        #state = np.reshape(ba, [-1, timesteps, num_input])
        feed_dict = {self.input_state: state}
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

    # training based on off policy ( td with monte carlo learning)
<<<<<<< HEAD
    def train_state_value(self, state, target_vs, from_memory=False):
        if from_memory:
            state, target_vs = self.memory.sample_batch(batch_size)
=======
    def train_state_value(self, state, target_vs):
        #st, at, rt, st1, advantage, state_value, done = memory.sample_batch(batch_size)
>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d
        st = np.reshape(state, [-1, state_space])
        rt = np.reshape(target_vs, [-1, rewards_space])
        feed_dict = {self.input_state: st,
                     self.target_vs: rt}
        self.sess.run(self.train_sv, feed_dict=feed_dict)
        return self.sess.run(self.vs_loss, feed_dict=feed_dict)

    def get_summary(self):
<<<<<<< HEAD
        st, state_value = self.memory.sample_batch(batch_size)
        st = np.reshape(st, [-1, state_space])
        state_value = np.reshape(state_value, [-1, rewards_space])
        feed_dict = {self.input_state: st,
                     self.target_vs: state_value}
        return self.sess.run(self.merged, feed_dict=feed_dict)

    def train_policy(self, state, action, value):
        st = np.reshape(state, [-1, state_space])
        act = np.reshape(action, [-1, action_space])
        val = np.reshape(value, [-1, rewards_space])
        feed_dict = {self.input_state: st,
=======
        st, at, rt, st1, advantage, state_value, done = self.memory.sample_batch(batch_size)
        feed_dict = {self.input_state: st,
                     self.target_vs: rt}
        return self.sess.run(self.merged, feed_dict=feed_dict)

    def train_policy(self, state, action, value):
        st = np.reshape(state, [-1, state_space])
        act = np.reshape(action, [-1, action_space])
        val = np.reshape(value, [-1, rewards_space])
        feed_dict = {self.input_state: st,
>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d
                     self.input_action: act,
                     self.target_vs: val}
        self.sess.run(self.train_pi, feed_dict=feed_dict)

    def compute_returns(self, states):
        buffer_v_target = []

        # Monte carlo estimates
<<<<<<< HEAD
        v_s_ = np.asarray([0.0])
=======
        v_s_ = np.asarray(0.0)
>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d
        for st in states[::-1]:
            v_s_ = np.add(st.reward_gotten, np.multiply(0.9, v_s_))
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        # update states and populate memory
        for i in range(0, len(buffer_v_target)):
            states[i].update_reward_value(np.reshape(buffer_v_target[i], [1, rewards_space]))
            # advantage = Rt+0.9*v_s(t+1) - v_s(t)
            advantage = np.subtract(buffer_v_target[i], states[i].state_value)
            states[i].update_advantage_value(advantage)
            self.memory.add(states[i])

    def run(self, saver, writer):
        train_step = 0
        total_step = 0
        # pull trained global net vars to local net
        self.pull_global()
        buffer_s, buffer_a, buffer_r = [], [], []
        while self.episodes < self.game_env.max_episodes:
            self.game_env.env.reset()
            done, steps, net_rewards, states = False, 0, 0.0, list()
            action = self.game_env.env.action_space.sample()
            st, _, _, _ = self.game_env.env.step(action)
            st = np.reshape(np.asarray(st), [1, self.game_env.state_space])

            # run one episode
            for ep_t in range(self.game_env.max_ep_step):
                self.game_env.env.render()
                at = self.get_action(st)
                st1, rt1, done, info = self.game_env.env.step(at)
                st1 = np.reshape(st1, [1, state_space])

                done = True if ep_t == self.game_env.max_ep_step - 1 else False
                net_rewards += rt1

                buffer_s.append(st)
                buffer_a.append(at)
                buffer_r.append((rt1 + 8) / 8)

<<<<<<< HEAD
                steps += 1

                if (total_step % self.game_env.max_ep_per_train == 0) or done:

=======

                steps += 1

                if (total_step % self.game_env.max_ep_per_train == 0) or done:

>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.get_state_value(state=st)

                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + 0.9 * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

<<<<<<< HEAD
                    for i in range(0, len(buffer_s)):
                        self.memory.add((buffer_s[i], buffer_v_target[i]))

                    vs_loss = self.train_state_value(buffer_s, buffer_v_target)

                    self.train_policy(buffer_s, buffer_a, buffer_v_target)
=======
                    vs_loss = self.train_state_value(buffer_s, buffer_v_target)

                    self.train_policy(buffer_s, buffer_a, buffer_v_target)

                    self.memory.add((buffer_s, buffer_a, buffer_v_target))
>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d

                    # discard episodes used for training pi
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # pull trained global net vars to local net
                    self.pull_global()
                    train_step += 1
                    print("training step {}: loss {}, avg_rewards {}".format(train_step, vs_loss,
                                                                             np.mean(self.avg_rewards)))
                
                st = st1
                total_step += 1
<<<<<<< HEAD
                self.train_state_value(None, None, from_memory=True)
=======
>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d

            # -- finished one episode --
            self.episodes += 1

            # computer running average rewards
            self.avg_rewards.append(net_rewards)
            if self.episodes > 10:
                self.avg_rewards.pop(0)

            summary = self.get_summary()
            writer.add_summary(summary, self.episodes)

            if self.episodes % 100 == 0:
                save_path = saver.save(self.sess, MODEL_PATH + "pretrained.ckpt", global_step=train_step)
                print("saved to %s" % save_path)

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, tr):
        experience = tr
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, sample_size):
        batch = []

        if self.count < sample_size:
            batch = random.sample(self.buffer, self.count)
        else:
            # with prob of 0.6 or more sample most recent events
            if random.uniform(0, 1) < 0.4:
                batch = random.sample(self.buffer, sample_size)
<<<<<<< HEAD

            else:
                batch = random.sample(list(self.buffer)[len(self.buffer)/2:], sample_size)

        states = np.reshape([_[0] for _ in batch], [-1, state_space])
        values = np.reshape([_[1] for _ in batch], [-1, rewards_space])

        return states, values
=======
            else:
                batch = random.sample(list(self.buffer)[len(self.buffer)/2:], sample_size)

        return batch
>>>>>>> f6643b28cbc77d16b4898f0547f099ecbfbad79d


def main():
    sess = tf.InteractiveSession()
    gm = game(sess)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG, sess.graph)

    checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    gm.run(saver, writer)


if __name__ == "__main__":
    main()
