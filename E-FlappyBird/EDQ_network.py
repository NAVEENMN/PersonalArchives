#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import model_network as model_net
import random
from numpy import interp
import numpy as np
from collections import deque

ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 60 # size of minibatch
FRAME_PER_ACTION = 1

class environment():
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.epsilon = INITIAL_EPSILON
        self.env = game.GameState()
        net = model_net.network()
        fam = model_net.familiarity()
        self.encoder_ph = fam.get_place_holders()
        self.latent_loss, self.kl_train = fam.train_network()
        self.plot = model_net.graphs()
        self.input_ph, self.a_ph, self.y_ph = net.get_place_holders()
        self.readout = net.get_read_out()
        self.train_step = net.train_network()
        self.setup_checkpoint()
        self.initialize()
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        
    def get_game_envirnoment(self):
        return self.env
    
    def setup_checkpoint(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks/")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
    
    def save_check_point(self, time_step):
        self.saver.save(self.sess, 'saved_networks/', global_step=time_step)
    
    def take_action(self, state, time_step):
        env = self.env
        a_t = np.zeros([ACTIONS])
        action_index = 0
        state = np.reshape(state, [1, 80, 80, 4])
        readout_t = self.feed_forward(state=state)[0]
        if time_step % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[action_index] = 1  # do nothing

        # scale down epsilon
        if self.epsilon > FINAL_EPSILON and time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
        action = a_t
        state, reward, terminal = env.frame_step(action)
        state = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        state = np.reshape(state, (80, 80, 1))
        return state, reward, action, terminal
    
    def feed_forward(self, state):
        readout_t = self.readout.eval(feed_dict={self.input_ph: state})
        return readout_t
    
    def check_feelings(self, state, avg_reward):
        latent_loss = self.latent_loss.eval(feed_dict={self.encoder_ph: [state]})
        latent_loss = interp(latent_loss, [0, 0.001], [0, 1])
        feed_dict = {self.encoder_ph: [state]}
        self.kl_train.run(feed_dict=feed_dict)
        scared = latent_loss
        courage = 1 - scared
        sad = interp(latent_loss, [0, 0.01], [0, 1])
        happy = sad-1
        data = [scared, courage, happy, sad]
        self.plot.plot(data=data)
        
    
    def train(self, transistion_history):
        D = transistion_history
        minibatch = random.sample(D, BATCH)
        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]
        
        y_batch = []
        s_j1_batch = np.reshape(s_j1_batch, [len(minibatch), 80, 80, 4])
        s_j_batch = np.reshape(s_j_batch, [len(minibatch), 80, 80, 4])
        readout_j1_batch = self.feed_forward(state=s_j1_batch)
        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
        feed_dict = {self.y_ph: y_batch, self.a_ph: a_batch, self.input_ph: s_j_batch}
        self.train_step.run(feed_dict=feed_dict)
    
def playGame(env):
    game = env.get_game_envirnoment()
    action = np.zeros(ACTIONS)
    action[0] = 1 # do nothing
    x_t, r_0, terminal = game.frame_step(action)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    D = deque()
    time_step=0
    counter = 0
    episode = 0
    total_reward = 0
    avg_reward = 0
    while True:
        counter +=1
        #if time_step % 10 == 0:
        #env.check_feelings(state=s_t, avg_reward=avg_reward)
        state, r_t, a_t, terminal = env.take_action(state=s_t, time_step=time_step)
        total_reward += r_t
        if terminal:
            episode +=1
            avg_reward = total_reward / counter
            counter = 0
            total_reward = 0
            print("EPISODE ", episode, "AVG_REWARD", avg_reward)
            
        s_t1 = np.append(state, s_t[:, :, :3], axis=2)
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if time_step > OBSERVE:
           env.train(transistion_history=D)
        s_t = s_t1
        time_step += 1

        if time_step % 10000 == 0:
            env.save_check_point(time_step=time_step)

def main():
    env = environment()
    playGame(env)

if __name__ == "__main__":
    main()
