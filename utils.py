import os
import pickle
import numpy as np
from PIL import Image
from collections import deque

'''
bGraph expects input as jpg
run images through bGraph and save in pickel binaries
'''
def process_bGraph_images(bGraph, DATA_PATH):
    total_samples = 0
    dirs = os.listdir(DATA_PATH)
    data_dic = dict()
    bneck_images = deque()
    number_of_classes = len(dirs)
    bottle_neck_pickel_path = DATA_PATH+'bottle_necks.pickle'

    # check if pickeled data exists if exists skip reprocessing
    if os.path.isfile(bottle_neck_pickel_path):
        print("loading data from pickel")
        with open(bottle_neck_pickel_path, 'rb') as handle:
            data = pickle.load(handle)
        return data["data"]

    # treating dir name as class names
    for cls, cls_name in enumerate(dirs):
        print("loading data for class {}: {}".format(cls, cls_name))

        path = DATA_PATH+cls_name
        cnt = 0
        for filename in os.listdir(path):
            img_path = path+"/"+filename
            print(img_path)
            image = Image.open(img_path)
            # bneck has a shape of [1, 1, 1, 2048]
            bneck = bGraph.get_bottle_neck_out(image)
            bneck = np.reshape(bneck, [-1, bneck.shape[-1]])
            _class = np.zeros((1, number_of_classes))
            _class[0][cls] = 1.0
            #print(bneck, _class)
            bneck_images.append((bneck, _class))
            total_samples += 1
            cnt += 1
            if cnt == 20:
                break
    data_dic["data"] = bneck_images

    with open(bottle_neck_pickel_path, 'wb') as handle:
        pickle.dump(data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("processed data saved ", bottle_neck_pickel_path)
    print("total samples", total_samples)

    return data_dic["data"]

