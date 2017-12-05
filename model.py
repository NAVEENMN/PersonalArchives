import tensorflow as tf

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 1
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
BATCH_SHAPE = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]

def weights_bias(shape, name):
  weight = tf.get_variable(name+"_W", shape)
  bias = tf.get_variable(name+"_B", [shape[len(shape)-1]])
  return weight, bias
def conv2d(x, W, stride):
  strides = [1, stride, stride, 1]
  return tf.nn.conv2d(x, W, strides=strides, padding="SAME")
def max_pool_2x2(x):
  strides = [1, 2, 2, 1]
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=strides, padding="SAME")
def feed_conv(self, input, weight, bias, stride):
  temp = tf.nn.relu(conv2d(input, weight, stride) + bias)
  return max_pool_2x2(temp)

class encoder_decoder():
  def __init__(self):
    #==== Encoder ====#
    with tf.variable_scope('Encoder_ConvNet'):
      self.E_conv1_W, self.E_conv1_B = weights_bias([5, 5, 1, 32], "E_conv1")
      self.E_conv2_W, self.E_conv2_B = weights_bias([5, 5, 32, 64], "E_conv2")
      self.E_conv3_W, self.E_conv3_B = weights_bias([5, 5, 64, 128], "E_conv3")
      self.E_fc1_W, self.E_fc1_B = weights_bias([20*20*128, 1024], "E_fc1")
      self.E_fc2_W, self.E_fc2_B = weights_bias([1024, 512], "E_fc2")
    #==== Decoder ====#
    with tf.variable_scope('Decoder_ConvNet'):
      self.D_fc1_W, self.D_fc2_B = weights_bias([512, 160*160*4], "D_fc1")
      self.D_conv1_W, self.D_conv1_B = weights_bias([3, 3, 1, 64], "D_conv1")
      self.D_conv2_W, self.D_conv2_B = weights_bias([3, 3, 64, 32], "D_conv2")
      self.D_conv3_W, self.D_conv3_B = weights_bias([1, 1, 32, 1], "D_conv3")
  
  def encode(self, image):
    with tf.variable_scope('Encoder_ConvNet'):
      input_st = tf.image.rgb_to_grayscale(image)
      h_conv1 = self.feed_conv(image, self.E_conv1_W, self.E_conv1_B, 1)
      h_conv2 = self.feed_conv(h_conv1, self.E_conv2_W, self.E_conv2_B, 1)
      h_conv3 = self.feed_conv(h_conv2, self.E_conv3_W, self.E_conv3_B, 1)
      h_conv3_flat = tf.contrib.layers.flatten(h_conv3)
      h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.E_fc1_W) + self.E_fc1_B)
      z = tf.matmul(h_fc1, self.E_fc2_W) + self.E_fc2_B
      return z

  def decode(self, latent):
    with tf.variable_scope('Decoder_ConvNet'):
      D_fc1 = tf.matmul(latent, self.D_fc1_W) + self.D_fc2_B
      D_fc1 = tf.reshape(D_fc1, [-1, IMAGE_WIDTH*2, IMAGE_HEIGHT*2, 1])
      D_fc1 = tf.contrib.layers.batch_norm(D_fc1, epsilon=1e-5, scope='bn')
      D_fc1 = tf.nn.relu(D_fc1)
      h_dconv1 = conv2d(D_fc1, self.D_conv1_W, 1) + self.D_conv1_B
      h_dpool1 = tf.contrib.layers.batch_norm(h_dconv1, epsilon=1e-5, scope='bn1')
      h_dpool1 = tf.nn.relu(h_dpool1)
      h_dconv1 = tf.image.resize_images(h_dpool1, [IMAGE_WIDTH*2, IMAGE_HEIGHT*2]
      h_dconv2 = conv2d(h_dconv1, self.D_conv2_W, 1) + self.D_conv2_B
      h_dpool2 = tf.contrib.layers.batch_norm(h_dconv2, epsilon=1e-5, scope='bn2')
      h_dpool2 = tf.nn.relu(h_dpool2)
      h_dconv2 = tf.image.resize_images(h_dpool2, [IMAGE_WIDTH*2, IMAGE_HEIGHT*2])
      h_dconv3 = conv2d(h_dconv2, self.D_conv3_W, 2) + self.D_conv3_B
      image = tf.nn.sigmoid(h_dconv3)
      return image

  def get_loss(self, source, target):
      with tf.name_scope('Loss'):
        source = tf.image.rgb_to_grayscale(source)
        batch_flatten = tf.reshape(source, [10, -1])
        batch_reconstruct_flatten = tf.reshape(target, [10, -1])
        loss1 = batch_flatten*tf.log(1e-10+batch_reconstruct_flatten)
        loss2 = (1 - batch_flatten) * tf.log(1e-10 + 1 - batch_reconstruct_flatten)
        loss = loss1+loss2
        reconstruction_loss = -tf.reduce_sum(loss)
        return tf.reduce_mean(reconstruction_loss)

   def train_step(self, loss)
       return tf.train.AdamOptimizer(0.001).minimize(loss)
