import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.contrib import rnn

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape = shape)
  return tf.Variable(initial)

class AC():
  def __init__(self):
    #Actor
    seqlen = 2
    dim = 3
    self.input_ph_actor = tf.placeholder("float", [None, 3])
    self.input_ph_td_error = tf.placeholder("float", [None, 1])
    W_fc1 = weight_variable([3, 3])
    b_fc1 = bias_variable([3])

    W_fc2 = weight_variable([3, 2])
    b_fc2 = bias_variable([2])
    W_fc3 = weight_variable([2, 1])
    b_fc3 = bias_variable([1])
    hfc1 = tf.matmul(self.input_ph_actor, W_fc1) + b_fc1
    hfc2 = tf.matmul(hfc1, W_fc2) + b_fc2
    self.action = tf.sigmoid(tf.matmul(hfc2, W_fc3) + b_fc3)
    actor_loss = -tf.log(self.action)*self.input_ph_td_error
    cost_actor = tf.reduce_mean(tf.squared_difference(actor_loss, self.action))
    self.train_step_actor = tf.train.AdamOptimizer(0.001).minimize(cost_actor)

    #Critic
    self.input_ph_critic = tf.placeholder("float", [None, 3])
    W_fc4 = weight_variable([3, 2])
    b_fc4 = bias_variable([2])
    W_fc5 = weight_variable([2, 1])
    b_fc5 = bias_variable([1])

    hfc3 = tf.matmul(self.input_ph_critic, W_fc4) + b_fc4
    self.value = tf.matmul(hfc3, W_fc5) + b_fc5
    td = tf.multiply(self.input_ph_td_error, 0.9)
    new_v = tf.add(td, self.value)
    cost_critic = tf.reduce_mean(tf.squared_difference(new_v, self.value))
    self.train_step_critic = tf.train.AdamOptimizer(0.001).minimize(cost_critic)

  def get_action(self, state, time_step):
    state = np.reshape(state, [-1, 3])
    feed_dict={self.input_ph_actor : state}
    action = self.action.eval(feed_dict = feed_dict)
    print("action", action)
    return action

  def update(self, s_j_batch, r_batch, s_j1_batch):
    v_st = self.value.eval(feed_dict={self.input_ph_critic : s_j_batch})
    v_st1 = self.value.eval(feed_dict={self.input_ph_critic : s_j1_batch})
    td_error  = (r_batch+0.9*v_st1) - v_st
    for _ in range(0, 10):
      feed_dict={self.input_ph_actor : s_j_batch, self.input_ph_td_error: td_error}
      self.train_step_actor.run(feed_dict = feed_dict)
    for _ in range(0, 10):
      feed_dict={self.input_ph_critic: s_j_batch, self.input_ph_td_error: td_error}
      self.train_step_critic.run(feed_dict = feed_dict)

class game():
  def __init__(self):
    self.env = gym.make('Pendulum-v0')
    self.env.reset()
    self.AC = AC()
    self.sess  = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())
    self.D = deque()

  def update(self):
    minibatch = random.sample(self.D, 50)

    s_j_batch = [d[0] for d in minibatch]
    a_batch = [d[1] for d in minibatch]
    r_batch = [d[2] for d in minibatch]
    s_j1_batch = [d[3] for d in minibatch]
    s_j_batch = np.reshape(s_j_batch, [50, 3])
    a_batch = np.reshape(a_batch, [50, 1])
    r_batch = np.reshape(r_batch, [50, 1])
    s_j1_batch = np.reshape(s_j1_batch, [50, 3])
    
    self.AC.update(s_j_batch, r_batch, s_j1_batch)

  def run_episodes(self):
    time_step = 0
    rewards = list()
    avg_reward = 0.0
    self.eps = 0.5
    states  = deque()
    st, _, _, _ = self.env.step(self.env.action_space.sample())
    st_1, _, _, _ = self.env.step(self.env.action_space.sample())
    states.append(st)
    states.append(st_1)
    while True:
      is_random = False
      self.env.render()
      action = self.AC.get_action(st, time_step)
      if random.random() < self.eps:
        action = self.env.action_space.sample()
        is_random = True
      st1, rt1, done, _ = self.env.step(action)
      rewards.append(rt1)
      if not is_random:
        st = np.reshape(st, [1, 3])
        st1 = np.reshape(st1, [1, 3])
        sars = [st, action, rt1, st1]
        states.popleft()
        states.append(st)
        #sars = [states, action, rt1, st1]
        self.D.append(sars)
      if len(rewards) == 10:
        avg_reward = sum(rewards)/len(rewards)
        rewards = list()
      if len(self.D) > 300:
        self.D.popleft()
      if time_step > 500:
        self.update()
      st = st1
      time_step +=1
      if time_step %500 == 0:
        self.eps -= 0.01
      print(time_step, avg_reward, self.eps)

def main():
  gm = game()
  gm.run_episodes()

if __name__ == "__main__":
  main()
