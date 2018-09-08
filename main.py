import numpy as np
import tensorflow as tf
from models import *
from utils import *

tfd = tf.contrib.distributions

MODEL_PATH = "saved/"
latent = 32
IMAGE_SHAPE = [None, 28, 28, 1]
TARGET_SHAPE = [None, 30]
LATENT_SHAPE = [None, latent]
batch_size = 30
mnist = tf.keras.datasets.mnist

class network():
    def __init__(self, sess):
        self.sess = sess
        self.opt = tf.train.AdamOptimizer(0.01)

        with tf.name_scope("Inputs"):
            self.input_ph = tf.placeholder("float", IMAGE_SHAPE, name="input_ph")
            self.latent_ph = tf.placeholder("float", LATENT_SHAPE, name="latent_ph")
            self.target_ph = tf.placeholder(tf.int32, TARGET_SHAPE, name="target_ph")

        with tf.name_scope('encoder'):
            phi = encoder("encode_")
            self.mu, self.sd = phi.encode(self.input_ph)

        with tf.name_scope('latent_vec'):
            self.z = sample_gaussian(self.mu, self.sd)  # prior

        with tf.name_scope('decoder'):
            _phi = decoder("decode_")  # p(_image|z)
            self.image = _phi.decode(self.z)
            self.generate_image = _phi.decode(self.latent_ph, reuse=True)

        with tf.name_scope('loss'):
            # log(p(_image)): E_{z ~ q(z|xi)}( log(p(xi|z)) )
            # kl_div(p(z) || q(z|x_i))
            p_xi = crossEntropy(obs=self.image, actual=self.input_ph)
            kl_div = kullbackLeibler(mu=self.mu, log_sigma=self.sd)
            self.cost = tf.reduce_mean(p_xi+kl_div)

        with tf.name_scope("opt"):
            self.vae_train_step = self.opt.minimize(self.cost)

    def encode(self, data_in):
        data_in = np.reshape(data_in, [-1, 28, 28, 1])
        feed_dict = {self.input_ph: data_in}
        prediction = self.sess.run([self.mu, self.sd], feed_dict=feed_dict)
        return prediction

    def generate_a_image(self, latent_sample):
        latent_in = np.reshape(latent_sample, [-1, latent])
        feed_dict = {self.latent_ph: latent_in}
        return self.sess.run(self.generate_image, feed_dict=feed_dict)

    def train_step(self, data_in, target):
        data_in = np.reshape(data_in, [-1, 28, 28, 1])
        feed_dict = {self.input_ph: data_in, self.target_ph: target}
        self.sess.run(self.vae_train_step, feed_dict=feed_dict)
        loss = self.sess.run(self.cost, feed_dict=feed_dict)
        return loss

class data():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.train_batch_size = x_train.shape[0]
        self.test_batch_size = x_test.shape[0]
        self.x_train = list(x_train/255.0)
        self.x_test = list(x_test/255.0)
        self.y_train = np.reshape(y_train, [self.train_batch_size])
        self.y_test = np.reshape(y_test, [-1, self.test_batch_size])
        self.cur_batch_idx = 0

    def get_next_batch(self):
        self.cur_batch_idx += batch_size
        if self.cur_batch_idx > self.train_batch_size:
            self.cur_batch_idx = 0
        mini_batch_x = self.x_train[self.cur_batch_idx: self.cur_batch_idx+batch_size]
        mini_batch_y = self.y_train[self.cur_batch_idx: self.cur_batch_idx+batch_size]
        mini_batch_x = np.reshape(mini_batch_x, [batch_size, 28, 28, 1])
        mini_batch_y = np.reshape(mini_batch_y, [-1, batch_size])
        return mini_batch_x, mini_batch_y

def train_loop(sess, graph, dat, saver):
    for step in range(0, 1000):
        mini_batch_x, mini_batch_y = dat.get_next_batch()
        loss = graph.train_step(mini_batch_x, mini_batch_y)
        print("step {} loss {}".format(step, loss))
        if step % 200 == 0:
            saver.save(sess, MODEL_PATH + "pretrained.ckpt", global_step=step)

def test_vae(graph, disp):
    mu = 0
    sd = 1
    latent_sample = np.random.normal(mu, sd, latent)
    image = graph.generate_a_image(latent_sample)
    image = image * 255.0
    image = np.reshape(image, [28, 28])
    disp.update_display(image)

def main():
    dat = data()
    mode = 1 # 0: train , 1: test

    with tf.Session() as sess:
        graph = network(sess)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        checkpoint = tf.train.get_checkpoint_state(MODEL_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        if mode == 0:
            train_loop(sess, graph, dat, saver)
        else:
            disp = window(graph)
            disp.update_display()

    disp.close_window()

if __name__ == "__main__":
    main()