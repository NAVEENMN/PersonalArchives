# value estimates monte carlo
import gym
import math
import random
import numpy as np
import tensorflow as tf
from collections import deque

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape = shape)
  return tf.Variable(initial)

class network():
  def __init__(self):
    W_fc1 = weight_variable([4, 2])
    b_fc1 = bias_variable([2])

    # input layer
    self.input_ph = tf.placeholder("float", [None, 4])
    self.action_ph = tf.placeholder("float", [None, 2])
    self.returns_ph = tf.placeholder("float", [None, 1])

    # hidden layers
    h_fc1 = tf.matmul(self.input_ph, W_fc1) + b_fc1
    self.readout = h_fc1#tf.nn.sigmoid(h_fc1)

    q_predictions = tf.reduce_sum(tf.multiply(self.readout, self.action_ph), reduction_indices=[1])
    new_q = q_predictions + 0.9 * (tf.subtract(self.returns_ph, q_predictions))
    cost = tf.reduce_mean(tf.squared_difference(new_q, q_predictions))
    self.train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

  def get_place_holders(self):
    return self.input_ph, self.action_ph, self.returns_ph

  def get_read_out(self):
    return self.readout

  def get_train_step(self):
    return self.train_step

class game():
  def __init__(self):
    self.env = gym.make('CartPole-v0')
    self.sess = sess = tf.InteractiveSession()
    net = network()
    self.input_ph, self.action_ph, self.returns_ph = net.get_place_holders()
    self.read_out = net.get_read_out()
    self.train_step = net.get_train_step()
    self.env.reset()
    self.gamma = 0.9
    self.alpha = 0.0001
    self.memory = 5000
    self.batch_size = 100
    self.epoch = 10
    self.eps = 0.5
    self.w = np.random.rand(4,2)
    '''
    self.w = np.array([[ 1.12408131,  0.39517095],
                       [ 0.2102553,   0.72281251],
                       [ 0.93014011,  0.40174619],
                       [ 0.55152993,  0.58659949]])
    '''
    self.state_values = list()
    self.D = deque()
    self.counter = 0
    self.sess.run(tf.global_variables_initializer())

  def take_action(self, state):
    input_ph = self.input_ph
    read_out = self.read_out
    state = np.reshape(state, [1, 4])
    readout_t = read_out.eval(feed_dict={input_ph : state})[0]
    q_s = np.dot(state, self.w)
    explore = random.random()
    a = np.zeros([1, 2])
    if explore < self.eps:
      a_t = random.randint(0,1)
    else:
      a_t = np.argmax(readout_t)
    a[0][a_t] = 1
    return a_t

  def run_an_episode(self):
    time_step = 0
    states = list()
    actions = list()
    random_action = random.randint(0,1)
    state, reward, done, info = self.env.step(random_action)
    while not done:
      self.env.render()
      a = np.zeros([1, 2])
      a_t = self.take_action(state)
      state, reward, done, info = self.env.step(a_t)
      states.append(state)
      a[0][a_t] = 1
      actions.append(a)
      time_step += 1
    g_t = self.get_return_for_episode(states)
    for x in range(0, len(states)):
      self.D.append((states[x], actions[x], g_t[x]))
      if len(self.D) > self.memory:
        self.D.popleft()
    self.counter += 1
    if self.counter % 20 == 0:
      self.eps = self.eps - 0.001
    self.env.reset()
    return time_step

  def get_return_for_episode(self, states):
    g_v = list()
    n_s = len(states)
    for tr in range(0, n_s):
      state = states[tr][0]
      a_t = states[tr][1]
      g_t = 0.1
      for k in range(0, n_s):
        rw = 0.1
        if k == (n_s-1):
          rw = -10
        g_t = g_t+math.pow(0.9, k)*rw
      n_s = n_s - 1
      g_v.append(g_t)
    g_v = np.reshape(g_v, [-1, 1])
    return g_v

  # gradient update
  def optimize(self):
    input_ph = self.input_ph
    action_ph = self.action_ph
    read_out = self.read_out
    if len(self.D) < self.batch_size:
      return

    s_j_batch = [d[0] for d in minibatch]
    a_batch = [d[1] for d in minibatch]
    g_batch = [d[2] for d in minibatch]

    s_j_batch = np.reshape(s_j_batch, [self.batch_size, -1])
    a_batch = np.reshape(a_batch, [self.batch_size, -1])
    g_batch = np.reshape(g_batch, [self.batch_size, -1])

    q_predictions = tf.reduce_sum(tf.multiply(read_out, action_ph), reduction_indice=[1])
    new_q = q_predictions + 0.9 * (tf.subtract(gt_ph, q_predictions))
    cost = tf.reduce_mean(tf.subtract(new_q, q_predictions))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)


  def update(self):
    input_ph = self.input_ph
    a_ph = self.action_ph
    return_ph = self.returns_ph
    train_step = self.train_step

    if len(self.D) < self.batch_size:
      return
    minibatch = random.sample(self.D, self.batch_size)
    s_j_batch = [d[0] for d in minibatch]
    a_batch = [d[1] for d in minibatch]
    g_batch = [d[2] for d in minibatch]

    s_j_batch = np.reshape(s_j_batch, [self.batch_size, -1])
    a_batch = np.reshape(a_batch, [self.batch_size, -1])
    g_batch = np.reshape(g_batch, [self.batch_size, -1])

    feed_dict = {input_ph: s_j_batch, a_ph : a_batch, return_ph: g_batch}
    for _ in range(0, self.epoch):
      train_step.run(feed_dict = feed_dict)

def main():
  gm = game()
  for x in range(0, 1000):
    ts = gm.run_an_episode()
  for x in range(0, 10000):
    ts = gm.run_an_episode()
    gm.update()
    print("epi", x, ts)

if __name__ == "__main__":
  main()
