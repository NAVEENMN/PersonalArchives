import csv
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from pandas.io.parsers import read_csv
from tensorflow.contrib.tensorboard.plugins import projector

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
IMAGE_CHANNEL = 1
SHAPE_IM = [None, 96, 96, 1]
SHAPE_CLASS = [None, 30]

#Filter sizes
Filter_1 = 8
F2 = 5
F3 = 3

BATCH_SIZE = 10
TRAIN_DATA_PATH = "data/training.csv"
TEST_DATA_PATH = "data/test.csv"
LOG_DIR = "Model/"

class cnn():
    def __init__(self, name):
        self.name = name
        self.learning_rate = 0.0001

    def feed_forward(self, input_image, reuse):
        n = self.name
        with tf.variable_scope(n, reuse=reuse):
            conv1 = tf.layers.conv2d(input_image, 32, Filter_1, activation=tf.nn.relu, name=n+"conv1")
            conv1 = tf.contrib.layers.batch_norm(conv1, epsilon=1e-5, scope='conv1_bn')
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name=n+"conv1_pool")
            conv2 = tf.layers.conv2d(conv1, 64, F2, activation=tf.nn.relu, name=n+"conv2")
            conv2 = tf.contrib.layers.batch_norm(conv2, epsilon=1e-5, scope='conv2_bn')
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name=n+"conv2_pool")
            conv3 = tf.layers.conv2d(conv2, 128, F3, activation=tf.nn.relu, name=n + "conv3")
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2, name=n + "conv3_pool")
            h_conv2_flat = tf.contrib.layers.flatten(conv3)
            fc1 = tf.layers.dense(h_conv2_flat, 256, activation=tf.nn.relu, name=n + "E_fc1")
            pred = tf.layers.dense(fc1, 30, activation=tf.nn.tanh, name=n + "E_fc2")
            return pred, conv1

    def get_loss(self, prediction, target):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=prediction, labels=target))
        return loss

    def get_kernel(self, layer_name):
        with tf.variable_scope(layer_name, reuse=True):
            weights = tf.get_variable("kernel")
        return weights

    def train_step(self, loss):
        lr = self.learning_rate
        tvars = tf.trainable_variables()

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            train_step = tf.train.AdamOptimizer(lr).minimize(loss=loss, var_list=tvars)
        return train_step


class network():
    def __init__(self, sess):
        self.sess = sess
        data = dict()
        with tf.name_scope("Inputs"):
            self.input_train_image = tf.placeholder("float", SHAPE_IM, name="input_train")
            self.input_test_image = tf.placeholder("float", SHAPE_IM, name="input_test")
            self.input_boxes = tf.placeholder("float", [None, 15, 4], name="input_boxes")
            self.train_target_class = tf.placeholder("float", SHAPE_CLASS, name="target_class")
            self.test_target_class = tf.placeholder("float", SHAPE_CLASS, name="target_test_class")

        with tf.name_scope("network"):
            self.net = cnn(name="cnn_")
            self.prediction, self.act = self.net.feed_forward(self.input_train_image, False)

            self.train_loss = self.net.get_loss(self.prediction,
                                                self.train_target_class)
            self.kernels_cn1 = self.net.get_kernel("cnn_/cnn_conv1")
            self.kernels_cn2 = self.net.get_kernel("cnn_/cnn_conv2")

            self.test_pred, self.testact = self.net.feed_forward(self.input_test_image, True)

            self.pixel_cor = tf.add(tf.multiply(self.test_pred, 48.0), 48.0)
            self.pixel_loc = tf.reshape(self.pixel_cor, [BATCH_SIZE, 30])

            self.tagged = tf.image.draw_bounding_boxes(self.input_test_image,
                                                       boxes=self.input_boxes)

            data["source_image"] = self.input_train_image
            data["test_images"] = self.tagged
            data["train_loss"] = self.train_loss
            data["train_act"] = self.act
            data["kernels_1"] = self.kernels_cn1
            data["kernels_2"] = self.kernels_cn2

            self.cnn_train = self.net.train_step(self.train_loss)

        with tf.name_scope("Tensorboard"):
            tensorboard_summary(data)
            self.merged = tf.summary.merge_all()

    def predict(self, data):
        data = np.reshape(data, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
        feed_dict = {self.input_train_image: data}
        predicted_classes = self.sess.run(self.prediction, feed_dict=feed_dict)
        return predicted_classes

    def get_loss(self, train, data_test):

        data_test = np.reshape(data_test, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
        feed_dict = {self.input_test_image: data_test}
        cord = self.sess.run(self.pixel_loc, feed_dict=feed_dict)
        boxes = self.get_bounding_boxes(data_test, cord)

        data_train = np.reshape(train[0], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
        target_train = np.reshape(train[1], [-1, 30])
        feed_dict = {self.input_train_image: data_train,
                     self.train_target_class: target_train,
                     self.input_test_image: data_test,
                     self.input_boxes: boxes}
        ops = [self.train_loss, self.merged]
        train_loss, summary = self.sess.run(ops, feed_dict=feed_dict)
        return train_loss, summary

    def train_step(self, data, target):
        data = np.reshape(data, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
        target = np.reshape(target, [-1, 30])
        feed_dict = {self.input_train_image: data, self.train_target_class: target}
        train = self.sess.run(self.cnn_train, feed_dict=feed_dict)
        return train

    def print_kernel(self):
        print("layers")
        tvars = tf.trainable_variables()
        for var in tvars:
            print(var)
        print(" ")

    def get_bounding_boxes(self, images, allpoints):
        # format [y_min, x_min, y_max, x_max]
        coll = []
        box_w_h = 1
        for i in range(0, len(images)):
            facepoints = allpoints[i]
            xp, yp = facepoints[0::2], facepoints[1::2]
            xp = np.reshape(xp, [15, 1])
            yp = np.reshape(yp, [15, 1])
            x_max, y_max = (xp+box_w_h)/IMAGE_WIDTH, (yp+box_w_h)/IMAGE_HEIGHT
            x_min, y_min = (xp-box_w_h)/IMAGE_WIDTH, (yp-box_w_h)/IMAGE_HEIGHT
            boxes = np.hstack((y_min, x_min, y_max, x_max))
            coll.append(boxes)
        return coll

    def train(self, train_data, test_data, writer, test_sample, saver):
        imgid = 0
        self.print_kernel()
        random.shuffle(train_data)
        for epoch in range(0, 10000):
            data_train = random.sample(train_data, BATCH_SIZE)
            data_test = random.sample(test_data, BATCH_SIZE)
            train_image_batch = np.reshape([_[0] for _ in data_train], [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
            train_target_batch = np.reshape([_[1] for _ in data_train], [BATCH_SIZE, 30])
            self.train_step(train_image_batch, train_target_batch)
            if epoch % 10 == 0:
                train = [train_image_batch, train_target_batch]
                train_loss, summary = self.get_loss(train, data_test)
                writer.add_summary(summary, epoch)
                print(epoch, train_loss)

                # for gif generation
                '''
                test_sample = np.reshape(test_sample, [1, 96, 96, 1])
                out = self.sess.run(self.pixel_cor, feed_dict={self.input_test_image: test_sample})
                out = np.reshape(out, [1, 30])
                out = out[0]
                xp, yp = out[0::2], out[1::2]
                xp = np.reshape(xp, [15, 1])
                yp = np.reshape(yp, [15, 1])
                x_max, y_max = (xp + 1) / IMAGE_WIDTH, (yp + 1) / IMAGE_HEIGHT
                x_min, y_min = (xp - 1) / IMAGE_WIDTH, (yp - 1) / IMAGE_HEIGHT
                box = np.hstack((y_min, x_min, y_max, x_max))
                box = np.reshape(box, [1, 15, 4])
                tagged = self.sess.run(self.tagged, feed_dict={self.input_test_image: test_sample,
                                                               self.input_boxes: box})
                #print(box)
                tagged = np.reshape(tagged, [96, 96])
                im = Image.fromarray(np.uint8(tagged*255.0))
                im.save("images/"+str(imgid)+".png")
                imgid += 1
                '''

            if epoch % 40 == 0:
                save_path = saver.save(self.sess, LOG_DIR + "pretrained.ckpt", global_step=epoch)
                print("saved to %s" % save_path)

def tensorboard_summary(data):
    source_image = data["source_image"]
    test_images = data["test_images"]
    activations = data["train_act"]
    activations = tf.reshape(activations, [BATCH_SIZE*32, 44, 44, 1])
    kernels_1 = data["kernels_1"]
    kernels_2 = data["kernels_2"]
    kernels_1 = tf.reshape(kernels_1, [32, Filter_1, Filter_1, 1])
    kernels_2 = tf.reshape(kernels_2, [32 * 64, F2, F2, 1])
    with tf.name_scope("summary/images"):
        image_to_tb = source_image * 255.0
        test_images_to_tb = test_images * 255.0
        tf.summary.image('src', image_to_tb, 4)
        tf.summary.image('testimg', test_images_to_tb, 4)
        tf.summary.image('activ', activations, 10)
        tf.summary.image('kernels_cn1', kernels_1, 10)
        tf.summary.image('kernels_cn2', kernels_2, 10)
    with tf.name_scope("summary/losses"):
        tf.summary.scalar("train_loss", data["train_loss"])
        #tf.summary.scalar("test_loss", data["test_loss"])

def load_data():
    print("loading..")
    train_df = read_csv(os.path.expanduser(TRAIN_DATA_PATH))  # load pandas dataframe
    test_df = read_csv(os.path.expanduser(TEST_DATA_PATH))  # load pandas dataframe
    train_df['Image'] = train_df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    test_df['Image'] = test_df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    train_df = train_df.dropna()  # drop all rows that have missing values in them
    test_df = test_df.dropna()
    X = np.vstack(train_df['Image'].values) / 255.
    X = np.reshape(X, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    X = X.astype(np.float32)
    T = np.vstack(test_df['Image'].values) / 255.
    T = np.reshape(T, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    T = T.astype(np.float32)
    y = train_df[train_df.columns[:-1]].values
    y = np.reshape(y, (-1, 30))
    # to train the CNN scale y in range -1 to 1
    y = (y - 48) / 48  # scale target coordinates to [-1, 1]
    y = y.astype(np.float32)
    ty = test_df[test_df.columns[:-1]].values
    ty = (ty - 48) / 48
    ty = ty.astype(np.float32)
    #X, y = shuffle(X, y, random_state=42)  # shuffle train data
    return X, y, T, ty

def main():
    X, y, T, Ty = load_data()
    load_sample = T[0]
    print(T[0].shape)
    train_data = list(zip(X, y))
    sess = tf.InteractiveSession()
    net = network(sess)
    saver = tf.train.Saver()

    # For latent space visulization.
    N = 10000  # Number of items (vocab size).
    D = 200  # Dimensionality of the embedding.
    embedding_var = tf.Variable(tf.random_normal([N, D]), name='word_embedding')
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    projector.visualize_embeddings(writer, config)
    checkpoint = tf.train.get_checkpoint_state(LOG_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    net.train(train_data, T, writer, load_sample, saver)


if __name__ == "__main__":
    main()