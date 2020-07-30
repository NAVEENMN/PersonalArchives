import io
import os
import time
import json
import flask
import hashlib
import pymongo
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow_inference.utils import label_map_util
from  tensorflow_inference.utils import process_response
from flask import request, render_template, url_for

current_dir = os.path.dirname(os.path.abspath(__file__))
labels_path = os.path.join(current_dir, "tensorflow_inference", "models", "coco_label_map.pbtxt")
model_path = os.path.join(current_dir, "tensorflow_inference", "models", "coco_inference_graph.pb")

global access_key
app = flask.Flask(__name__, template_folder='template')


class database():
    def __init__(self):
        conn = "mongodb+srv://test_user:test@cluster0-onaoj.mongodb.net/test?retryWrites=true&w=majority"
        self.client = pymongo.MongoClient(conn)
        self.db = client.test

    def add_entry(self, data):
        post_id = db.posts.insert_one(data).inserted_id
        return post_id


class network():
    def __init__(self):
        # load graph, categories and labels
        self.label_map = label_map_util.load_labelmap(labels_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                            max_num_classes=90,
                            use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.graph = self.load_graph(model_path)
        # init session for this graph
        self.sess = tf.Session(graph=self.graph)

        # gather all tensors
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        #self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
        # gather ops
        self.ops = []
        self.ops.append(self.detection_boxes)
        self.ops.append(self.detection_scores)
        self.ops.append(self.detection_classes)

    def load_graph(self, path):
        # Load protobuf file from the disk and retrive unserialized graph
        with tf.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # import graph_def into a new graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

        return graph
    def run_inference(self, input_image):
        image = np.asarray(input_image)
        image = np.expand_dims(image, axis=0)
        feed_dict={self.image_tensor:image}
        return self.sess.run(self.ops,feed_dict=feed_dict)


net = network()
@app.route("/predict", methods=["POST"])
def predict():
    response = {"success": False}
    if flask.request.method == "POST":

        image_file = request.files['image']
        image_file.save("test.jpg")
        image = Image.open('test.jpg')

        start_time = time.time()
        preds = net.run_inference(image)
        results, total = process_response.process(net.category_index,
                    image, preds, draw_boxes=True, save_it=True)
        end_time = time.time() - start_time

        details = dict()
        details["predictions"] = results
        details["total_preds"] = total
        details["run_time"] = end_time

        response["success"] = True
        response["details"] = details

    return flask.jsonify(response)

@app.route("/get_key", methods=["POST"])
def get_key():
    response = {"success": False}
    if flask.request.method == "POST":
        in_data = json.loads(request.data)
        req_key = in_data["request_key"]
        if req_key == "give_me_access":
            global access_key
            response["success"] = True
            response["access_key"] = access_key
        else:
            response["err_msg"] = "access_denied"
        return flask.jsonify(response)

@app.route('/')
def root_page():
    return render_template("main.html")

@app.route('/image')
def image_page():
    return render_template("image.html")

def main():
    global access_key
    m = hashlib.new('ripemd160')
    m.update(('%s%s' % ("blue", "green")).encode('utf-8'))
    access_key = m.hexdigest()
    app.run(host="0.0.0.0", port=80)

if __name__ == "__main__":
    main()
