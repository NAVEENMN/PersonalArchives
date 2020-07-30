import tensorflow as tf

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
BATCH_SHAPE = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]


# ------ State encoder decoder -----#
class state_encoder_decoder():
    def __init__(self, name, batch_size):
        self.name = name
        self.learning_rate = 0.01
        self.latent_dim = 512
        self.batch_size = batch_size
        self.enc_name = "StEnc_"
        self.dec_name = "StDec_"

    def encode(self, image, reuse=False):
        n = self.enc_name
        with tf.variable_scope('Encoder_ConvNet', reuse=reuse):
            input_st = image
            conv1 = tf.layers.conv2d(input_st, 32, 5, activation=tf.nn.relu, name=n + "conv1")
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name=n + "conv1_pool")
            conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu, name=n + "conv2")
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name=n + "conv2_pool")
            conv3 = tf.layers.conv2d(conv2, 128, 5, activation=tf.nn.relu, name=n + "conv3")
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2, name=n + "conv3_pool")
            h_conv3_flat = tf.contrib.layers.flatten(conv3)
            fc1 = tf.layers.dense(h_conv3_flat, 1024, activation=tf.nn.sigmoid, name=n + "E_fc1")
            z = tf.layers.dense(fc1, self.latent_dim, activation=tf.nn.sigmoid, name=n + "E_fc2")
            return z

    def decode(self, latent, reuse=False):
        n = self.dec_name
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
            image = tf.layers.conv2d(dconv2, 3, 1, activation=tf.nn.sigmoid, name=n + "conv3")
            return image

    def get_loss(self, source, target):
        with tf.name_scope('St_endecLoss'):
            batch_flatten = tf.reshape(target, [self.batch_size, -1])
            batch_reconstruct_flatten = tf.reshape(source, [self.batch_size, -1])
            loss1 = batch_flatten * tf.log(1e-10 + batch_reconstruct_flatten)
            loss2 = (1 - batch_flatten) * tf.log(1e-10 + 1 - batch_reconstruct_flatten)
            loss = loss1 + loss2
            reconstruction_loss = -tf.reduce_sum(loss)
            return tf.reduce_mean(reconstruction_loss)

    def train_step(self, loss):
        lr = self.learning_rate
        tvars = tf.trainable_variables()
        st_encoder_vars = [var for var in tvars if self.enc_name in var.name]
        st_decoder_vars = [var for var in tvars if self.dec_name in var.name]
        st_vars = st_encoder_vars + st_decoder_vars
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=st_vars)
        return train_step

# ------ Action encoder decoder -----#
class action_encode_decoder():
    def __init__(self, name, batch_size):
        self.name = name
        self.learning_rate = 0.01
        self.latent_dim = 512
        self.enc_name = "AtEnc_"
        self.dec_name = "ATDec_"

    def encode(self, action, reuse=False):
        n = self.enc_name
        with tf.variable_scope('Encoder_FcNet', reuse=reuse):
            fc1 = tf.layers.dense(action, 1024, name=n + "fc1")
            action_latent = tf.layers.dense(fc1, 512, activation=tf.nn.sigmoid, name=n + "fc2")
            return action_latent

    def decode(self, action_latent, reuse=False):
        n = self.dec_name
        with tf.variable_scope('Decoder_FcNet', reuse=reuse):
            fc1 = tf.layers.dense(action_latent, 512 / 2, name=n + "fc1")
            action = tf.layers.dense(fc1, 18, name=n + "fc2")
            return action

    def get_loss(self, source, target):
        with tf.name_scope('At_endecLoss'):
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=source, labels=target)
            return tf.reduce_mean(reconstruction_loss)

    def train_step(self, loss):
        lr = self.learning_rate
        tvars = tf.trainable_variables()
        at_encoder_vars = [var for var in tvars if self.enc_name in var.name]
        at_decoder_vars = [var for var in tvars if self.dec_name in var.name]
        at_vars = at_encoder_vars + at_decoder_vars
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=at_vars)
        return train_step

# ---- Critic ----#
class sr_representation():
    def __init__(self, name, batch_size):
        self.name = name
        self.layer = "Sr_"
        self.latent_dim = 512
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.alpha = 0.8

    def get_sucessor_feature(self, phi_st):
        n = self.layer
        with tf.variable_scope('Sucessor_FcNet'):
            fc1 = tf.layers.dense(phi_st, 512,  activation=tf.nn.sigmoid, name=n + "fc1")
            fc2 = tf.layers.dense(fc1, 512/2, activation=tf.nn.sigmoid, name=n + "fc2")
            sr_feature = tf.layers.dense(fc2, 512, activation=tf.nn.sigmoid, name=n + "fc3")
        return sr_feature

    def get_q_value(self, feature):
        n = self.layer
        with tf.variable_scope('Sucessor_Q_FcNet'):
            fc1 = tf.layers.dense(feature, 512 / 2, activation=tf.nn.sigmoid, name=n + "fc1")
            fc2 = tf.layers.dense(fc1, 512 / 4, activation=tf.nn.sigmoid, name=n + "fc2")
            qvalues = tf.layers.dense(fc2, 18,activation=tf.nn.sigmoid,  name=n + "fc3")
            return qvalues

    def sr_train_step(self, phi_st, si_st, si_st1):
        lr = self.learning_rate
        td_target = phi_st + (self.gamma * si_st1)
        td_error = tf.losses.mean_squared_error(labels=td_target, predictions=si_st)
        tvars = tf.trainable_variables()
        sr_vars = [var for var in tvars if self.layer in var.name]
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_step = tf.train.AdamOptimizer(lr).minimize(td_error, var_list=sr_vars)
        return train_step, td_error


class reward_predictor():
    def __init__(self, name, batch_size):
        self.name = name
        self.learning_rate = 0.01
        self.layer = "Rw_"

    def predict_reward(self, phi_st, reuse=False):
        n = self.layer
        with tf.variable_scope('Reward_FcNet', reuse=reuse):
            fc1 = tf.layers.dense(phi_st, 512/2, activation=tf.nn.sigmoid, name=n + "fc1")
            fc2 = tf.layers.dense(fc1, 512/4, activation=None, name=n + "fc2")
            reward = tf.layers.dense(fc2, 1, activation=None, name=n + "fc3")
            return reward

    def get_loss(self, source, target):
        loss = tf.losses.mean_squared_error(predictions=source, labels=target)
        return loss

    def rw_train_step(self, loss):
        lr = self.learning_rate
        tvars = tf.trainable_variables()
        #st_encoder_vars = [var for var in tvars if "StEnc_" in var.name]
        rw_vars = [var for var in tvars if self.layer in var.name]
        #rw_vars = rw_vars + st_encoder_vars
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, var_list=rw_vars)
        return train_step


def tensorboard_summary(data):
    st = data["source_image"]
    st_recon = data["reconstructed_image"]
    sr_feature_recon = data["sr_feature"]
    with tf.name_scope("summary/images"):
        source_image = st * 255.0  # (st+1) * 127.5
        recon_image = st_recon * 255.0  # (st_recon+1) * 127.5
        sr_feature_recon = sr_feature_recon * 255.0
        image_to_tb = tf.concat([source_image, recon_image], axis=1)
        image_to_tb = tf.concat([image_to_tb, sr_feature_recon], axis=1)
        tf.summary.image('src', image_to_tb, 5)
        # tf.summary.image('recon', recon_image, 5)
    with tf.name_scope("summary/losses"):
        tf.summary.scalar("StRecon_loss", data["reconstruction_loss"])
        tf.summary.scalar("RtPred_loss", data["reward_pred_loss"])
        tf.summary.scalar("SR_loss", data["sr_loss"])
        tf.summary.scalar("At_loss", data["at_loss"])

