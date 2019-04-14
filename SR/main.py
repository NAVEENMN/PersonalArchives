from defs import *
from models import *
from utils import *
import gym
import numpy as np
import tensorflow as tf

IMAGE_WIDTH, IMAGE_HEIGHT = 96, 96
IMAGE_CHANNEL = 3
ST_SHAPE = [1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
ST_SHAPES = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
MAX_EP_STEP = 200

class network():
    def __init__(self, sess):
        self.sess = sess

        with tf.name_scope("Inputs"):
            self.input_st_ph = tf.placeholder("float", ST_SHAPES, name="input_st")

        with tf.name_scope("Phi"):
            stendec = state_encoder_decoder("Phi")

            with tf.name_scope("Phi_Encode"):
                self.phi = stendec.encode(self.input_st_ph, False)


    def get_state_latent(self, input_st):
        input_st = np.reshape(input_st, ST_SHAPE)
        latent = self.sess.run(self.phi, feed_dict={self.input_st_ph: input_st})
        return latent


class game():
    def __init__(self, sess):
        self.sess = sess
        self.env = gym.make('CarRacing-v0')
        self.env.reset()
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.action_bond = [self.env.action_space.low, self.env.action_space.high]
        self.net = network(self.sess)
        self.sess.run(tf.global_variables_initializer())
            
    def run(self):
        done = False
        running_step = 0
        st = np.asarray(self.env.reset())
        while not done:
            self.env.render()
            at = self.env.action_space.sample()
            res = self.net.get_state_latent(st)
            print(res)
            print(at)
            st1, r, done, info = self.env.step(at)
            print(st1.shape, done)
            running_step += 1
            done = True if running_step == MAX_EP_STEP - 1 else False

def main():
    sess = tf.Session()
    gm = game(sess)
    gm.run()

    state = np.random.rand(1, 32, 32, 3)
    res = net.get_state_latent(state)
    print(res)


if __name__ == "__main__":
    main()
