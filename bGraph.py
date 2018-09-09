import os
import numpy as np
import tensorflow as tf

class bottle_neck_graph():
    def __init__(self, dir_path, sess):
        self.graph = self.load_graph(dir_path)
        self.sess = tf.Session(graph=self.graph)
        self.print_graph()
        self.input = self.graph.get_tensor_by_name('DecodeJpeg:0')
        self.bottle_neck = self.graph.get_tensor_by_name('pool_3:0')

    def print_graph(self):
        print("bGraph layers")
        tensors = []
        with tf.Session(graph=self.graph) as _sess:
            op = _sess.graph.get_operations()
            tensors = [m.values() for m in op]
            for tensor in tensors:
                print(tensor)

    def get_bottle_neck_out(self, image):
        bneck = self.sess.run(self.bottle_neck, feed_dict={self.input: image})
        return bneck

    def load_graph(self, path):
        # Load protobuf file from the disk and retrive unserialized graph
        with tf.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # import graph_def into a new graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph