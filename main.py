from defs import *
from models import *
import tensorflow as tf
import numpy as np
import gym

class network():
    def __init__(self, sess):
        self.sess = sess

        with tf.name_scope("Inputs"):
            self.input_st_ph = tf.placeholder("float", ST_SHAPES, name="input_st")
            self.input_st_latents = tf.placeholder("float", ST_LT_SHAPES, name="input_state_latents")

        with tf.name_scope("Phi"):
            stendec = state_encoder_decoder("Phi", LATENT_DIM)
            with tf.name_scope("Phi_Encode"):
                self.phi = stendec.encode(self.input_st_ph, False)
            with tf.name_scope("Phi_Decode"):
                self.st_ = stendec.decode(self.phi, False)
                self.st_recon = stendec.decode(self.input_st_latents, True)
            with tf.name_scope("Phi_loss"):
                self.phi_st_loss = stendec.get_loss(source=self.st_,
                                                    target=self.input_st_ph)
            with tf.name_scope("Phi_train"):
                self.phi_st_train_step = stendec.train_step(self.phi_st_loss)

    def get_state_latent(self, input_st):
        input_st = np.reshape(input_st, ST_SHAPE)
        latent = self.sess.run(self.phi, feed_dict={self.input_st_ph: input_st})
        return latent

    def get_state_reconstruction(self, state_latent):
        state_latent = np.reshape(state_latent, (-1, LATENT_DIM))
        recon = self.sess.run(self.st_recon, feed_dict={self.input_st_latents: state_latent})
        return recon

class game():
    def __init__(self, sess):
        self.sess = sess
        # Game environment
        self.env_name = 'Acrobot-v1'
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.reset()
        self.print_game_env()
        st, _, _, _ = self.env.step(self.env.action_space.sample())
        self.net = network(sess)
        self.sess.run(tf.global_variables_initializer())

    def print_game_env(self):
        print("-----")
        print("environment: {}".format(self.env_name))
        print("action space : {}".format(self.action_space))
        screen = self.env.render(mode='rgb_array')
        print("state space : {}".format(self.observation_space))
        print("observation space : {}".format(screen.shape))
        print("reward range : {}".format(self.env.reward_range))
        print("------\n")

    def run(self):
        at = self.env.action_space.sample()
        st, rt, done, _ = self.env.step(at)
        screen = self.env.render(mode='rgb_array')
        print(screen.shape)
        print("res", [st, at, rt])
        #im = np.random.rand(1, 96, 96, 3)
        #z = self.net.get_state_latent(im)
        #rec = self.net.get_state_reconstruction(z)
        #print(z)
        #print(rec)


def main():
    sess = tf.InteractiveSession()
    gm = game(sess)
    for _ in range(10):
        gm.run()

if __name__ == "__main__":
    main()