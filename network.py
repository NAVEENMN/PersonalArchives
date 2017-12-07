import tensorflow as tf

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
BATCH_SHAPE = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]

class state_encoder_decoder():
  def __init__(self, name, batch_size):
    self.name = name
    self.learning_rate = 0.0001
    self.latent_dim = 512
    self.batch_size = batch_size
    self.enc_name = "StEnc_"
    self.dec_name = "StDec_"
  
  def encode(self, image):
    n = self.enc_name
    with tf.variable_scope('Encoder_ConvNet'):
      input_st = image
      conv1 = tf.layers.conv2d(input_st,32,5, activation=tf.nn.relu, name=n+"conv1")
      conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name="E_conv1_pool")
      conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu, name=n+"conv2")
      conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name="E_conv2_pool")
      conv3 = tf.layers.conv2d(conv2, 128, 5, activation=tf.nn.relu, name=n+"conv3")
      conv3 = tf.layers.max_pooling2d(conv3, 2, 2, name="E_conv3_pool")
      h_conv3_flat = tf.contrib.layers.flatten(conv3)
      fc1 = tf.layers.dense(h_conv3_flat, 1024, name="E_fc1")
      z = tf.layers.dense(fc1, self.latent_dim, name="E_fc2")
      return z

  def decode(self, latent):
    n = self.dec_name
    with tf.variable_scope('Decoder_ConvNet'):
      fc1 = tf.layers.dense(latent, IMAGE_PIXELS*4, name=n+"fc")
      D_fc1 = tf.reshape(fc1, [-1, IMAGE_WIDTH*2, IMAGE_HEIGHT*2, IMAGE_CHANNEL])
      D_fc1 = tf.contrib.layers.batch_norm(D_fc1, epsilon=1e-5, scope='bn')
      D_fc1 = tf.nn.relu(D_fc1)
      dconv1 = tf.layers.conv2d(D_fc1, 64, 5, activation=tf.nn.relu, name=n+"conv1")
      dconv1 = tf.contrib.layers.batch_norm(dconv1, epsilon=1e-5, scope='bn1')
      dconv1 = tf.image.resize_images(dconv1, [IMAGE_WIDTH*2, IMAGE_HEIGHT*2])
      dconv2 = tf.layers.conv2d(dconv1, 32, 3, activation=tf.nn.relu, name=n+"conv2")
      dconv2 = tf.contrib.layers.batch_norm(dconv2, epsilon=1e-5, scope='bn2')
      dconv2 = tf.image.resize_images(dconv2, [IMAGE_WIDTH*1, IMAGE_HEIGHT*1])
      image  = tf.layers.conv2d(dconv2,3,1,activation=tf.nn.sigmoid, name=n+"conv3")
      return image

  def get_loss(self, source, target):
      with tf.name_scope('Loss'):
        #target = tf.image.rgb_to_grayscale(target)
        batch_flatten = tf.reshape(target, [self.batch_size, -1])
        batch_reconstruct_flatten = tf.reshape(source, [self.batch_size, -1])
        loss1 = batch_flatten*tf.log(1e-10+batch_reconstruct_flatten)
        loss2 = (1 - batch_flatten) * tf.log(1e-10 + 1 - batch_reconstruct_flatten)
        loss = loss1+loss2
        reconstruction_loss = -tf.reduce_sum(loss)
        return tf.reduce_mean(reconstruction_loss)
  
  def train_step(self, loss):
    lr = self.learning_rate
    tvars = tf.trainable_variables()
    st_encoder_vars = [var for var in tvars if self.enc_name in var.name]
    st_decoder_vars = [var for var in tvars if self.dec_name in var.name]
    st_vars = st_encoder_vars+st_decoder_vars
    with tf.variable_scope(tf.get_variable_scope()) as scope:
      train_step = tf.train.AdamOptimizer(lr).minimize(loss,var_list = st_vars)
    return train_step

def tensorboard_summary(data):
  st = data["source_image"]
  st_recon = data["reconstructed_image"]
  with tf.name_scope("summary/images"):
    source_image = st*255#(st+1) * 127.5
    recon_image  = st*255#(st_recon+1) * 127.5
    image_to_tb = tf.concat([source_image, recon_image], axis=1)
    tf.summary.image('src', source_image, 5)
    tf.summary.image('recon', recon_image, 5)
  with tf.name_scope("summary/losses"):
    tf.summary.scalar("StRecon_loss", data["reconstruction_loss"])
