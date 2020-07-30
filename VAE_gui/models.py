import tensorflow as tf

latent = 32
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNEL = 1
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL

def sample_gaussian(mu, sd):
    """Draw sample from Gaussian with given shape, subject to random noise epsilon"""
    with tf.name_scope("sample_gaussian"):
        epsilon = tf.random_normal([tf.shape(sd)[0], latent], name="epsilon")
        return mu + epsilon * tf.exp(sd) # N(mu, sigma**2)

def crossEntropy(obs, actual, offset=1e-7):
    """Binary cross-entropy, per training example"""
    with tf.name_scope("cross_entropy"):  # bound by clipping to avoid nan
        obs_ = tf.clip_by_value(obs, offset, 1 - offset)
        return -tf.reduce_sum(actual * tf.log(obs_)+ (1 - actual) * tf.log(1 - obs_), 1)

def kullbackLeibler(mu, log_sigma):
    """(Spherical Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
    with tf.name_scope("KL_divergence"):  # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
        return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu ** 2 - tf.exp(2 * log_sigma), 1)

class encoder():
    def __init__(self, name):
        self.name = name
        self.learning_rate = 0.01

    def encode(self, image, reuse=False):
        n = self.name
        with tf.variable_scope('Encoder_ConvNet', reuse=reuse):
            conv1 = tf.layers.conv2d(image, 32, 5, activation=tf.nn.relu, name=n+"conv1")
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name=n+"conv1_pool")
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name=n+"conv2")
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name=n+"conv2_pool")
            h_conv2_flat = tf.contrib.layers.flatten(conv2)
            mu = tf.layers.dense(h_conv2_flat, 1, activation=tf.nn.tanh, name=n+"fc1_mu")
            sd = tf.layers.dense(h_conv2_flat, 1, activation=tf.nn.relu, name=n+"fc1_sd")
        return mu, sd

class decoder():
    def __init__(self, name):
        self.name = name
        self.learning_rate = 0.01

    def decode(self, latent, reuse=False):
        n = self.name
        with tf.variable_scope('Decoder_ConvNet', reuse=reuse):
            fc1 = tf.layers.dense(latent, IMAGE_PIXELS * 4, name=n + "fc")
            D_fc1 = tf.reshape(fc1, [-1, IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2, IMAGE_CHANNEL])
            D_fc1 = tf.contrib.layers.batch_norm(D_fc1, epsilon=1e-5, scope='bn')
            D_fc1 = tf.nn.relu(D_fc1)
            dconv1 = tf.layers.conv2d(D_fc1, 64, 5, activation=tf.nn.relu, name=n + "conv1")
            dconv1 = tf.contrib.layers.batch_norm(dconv1, epsilon=1e-5, scope='bn1')
            dconv1 = tf.image.resize_images(dconv1, [IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2])
            dconv2 = tf.layers.conv2d(dconv1, 32, 3, activation=tf.nn.relu, name=n + "conv2")
            dconv2 = tf.contrib.layers.batch_norm(dconv2, epsilon=1e-5, scope='bn2')
            dconv2 = tf.image.resize_images(dconv2, [IMAGE_WIDTH * 1, IMAGE_HEIGHT * 1])
            image = tf.layers.conv2d(dconv2, 1, 1, activation=tf.nn.sigmoid, name=n + "conv3")
        return image