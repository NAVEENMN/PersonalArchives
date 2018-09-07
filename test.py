import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

BOTTLE_NECK_GRAPH_PATH = "pb_models/classify_image_graph_def.pb"
OUT_GRAPH_PATH = "pb_models/frozen_tGraph.pb"

dir = os.path.dirname(os.path.realpath(__file__))
test_img = dir+"/data/plastic/plastic11.jpg"

class network():
    def __init__(self, sess):
        self.sess = sess
        self.bGraph = self.load_graph(BOTTLE_NECK_GRAPH_PATH)
        self.tGraph = self.load_graph(OUT_GRAPH_PATH)
        self.input_bGraph_ph = self.bGraph.get_tensor_by_name('DecodeJpeg:0')
        self.bottle_neck = self.bGraph.get_tensor_by_name('pool_3:0')
        self.input_tGraph_ph = self.tGraph.get_tensor_by_name('input_ph:0')
        self.output_tensor = self.tGraph.get_tensor_by_name('FC_layers_1/FC_4/Softmax:0')

    def feed_forward(self, image):
        feed_dict = {self.input_bGraph_ph: image}
        with tf.Session(graph=self.bGraph) as _sess:
            bneck = _sess.run(self.bottle_neck, feed_dict=feed_dict)
        bneck = np.reshape(bneck, [-1, 2048])
        feed_dict = {self.input_tGraph_ph: bneck}
        with tf.Session(graph=self.tGraph) as _sess:
            out = _sess.run(self.output_tensor, feed_dict=feed_dict)
        return out

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
    image = Image.open(test_img)

    with tf.Session() as sess:
        graph = network(sess)
        start_time = time.time()
        output = graph.feed_forward(image)
        print(output[0], np.argmax(output[0]))
        run_time = time.time() - start_time
        print("--- %s seconds ---" % (run_time))

if __name__ == "__main__":
    main()