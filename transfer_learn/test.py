import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.python.client import device_lib

BOTTLE_NECK_GRAPH_PATH = "pb_models/classify_image_graph_def.pb"
OUT_GRAPH_PATH = "pb_models/frozen_tGraph.pb"

dir = os.path.dirname(os.path.realpath(__file__))
test_img = dir+"/data/foodboxes/1/ds1_f1_13.jpg"

class network():
    def __init__(self, sess):
        self.bGraph = self.load_graph(BOTTLE_NECK_GRAPH_PATH)
        self.bsess = tf.Session(graph=self.bGraph)
        self.tGraph = self.load_graph(OUT_GRAPH_PATH)
        self.tsess = tf.Session(graph=self.tGraph)
        self.input_bGraph_ph = self.bGraph.get_tensor_by_name('DecodeJpeg:0')
        self.bottle_neck = self.bGraph.get_tensor_by_name('pool_3:0')
        self.input_tGraph_ph = self.tGraph.get_tensor_by_name('input_ph:0')
        self.output_class = self.tGraph.get_tensor_by_name('FC_layers_1/pred_out/Softmax:0')
        self.output_probpick = self.tGraph.get_tensor_by_name('FC_layers_1/pick_out/Sigmoid:0')

    def feed_forward(self, image):
        feed_dict = {self.input_bGraph_ph: image}
        bneck = self.bsess.run(self.bottle_neck, feed_dict=feed_dict)
        bneck = np.reshape(bneck, [-1, 2048])
        feed_dict = {self.input_tGraph_ph: bneck}
        out, prob = self.tsess.run([self.output_class, self.output_probpick], feed_dict=feed_dict)
        return out, prob

    def load_graph(self, path):
        # Load protobuf file from the disk and retrive unserialized graph
        with tf.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # import graph_def into a new graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        return graph


def main():
    devs = device_lib.list_local_devices()
    image = Image.open(test_img)
    classes = ["cans", "foodboxes", "hdp", "nothing"]
    with tf.Session() as sess:
        with tf.device(devs[1].name):
            graph = network(sess)
            start_time = time.time()
            output, pick_prob = graph.feed_forward(image)
            idx = np.argmax(output[0])
            obj = classes[idx]
        print(output[0], obj, pick_prob)
        run_time = time.time() - start_time
        print("--- %s seconds ---" % (run_time))

if __name__ == "__main__":
    main()