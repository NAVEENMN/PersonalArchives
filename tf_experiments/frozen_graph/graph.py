'''
 load pb (protoBuf) file and run an inference
'''
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

img_path = "/Users/naveenmysore/Documents/me.jpg"

class graph():
    def __init__(self, dir_path, model_name):
        self.graph = self.load_graph(dir_path, model_name)
        self.input_ph = self.graph.get_tensor_by_name('DecodeJpeg:0')
        #self.mid = self.graph.get_tensor_by_name('add_6:0')
        self.output = self.graph.get_tensor_by_name('softmax:0')

    def load_graph(self, dir_path, model_name):
        path = os.path.join(dir_path, model_name)
        # Load protobuf file from the disk and retrive unserialized graph
        with tf.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # import graph_def into a new graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph

    def run_g(self, input_image):
        print("vars")
        feed_dict = {self.input_ph: input_image}
        with tf.Session(graph=self.graph) as sess:
            op = sess.graph.get_operations()
            tensors = [m.values() for m in op]
            for tensor in tensors:
                print(tensor)
            res = sess.run([self.output],feed_dict=feed_dict)
            print(res)


def main():
    gh = graph("model", "classify_image_graph_def.pb")
    image = Image.open(img_path)
    #image = np.random.rand(1, 32, 32, 3)
    gh.run_g(image)

if __name__ == "__main__":
    main()
