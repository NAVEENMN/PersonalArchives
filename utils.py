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
        print("loading data from pickel", bottle_neck_pickel_path)
        with open(bottle_neck_pickel_path, 'rb') as handle:
            data = pickle.load(handle)
            print(data)
        return data["data"]

    # treating dir name as class names
    for cls, cls_name in enumerate(dirs):
        print("loading data for class {}: {}".format(cls, cls_name))
        path = DATA_PATH+cls_name
        cnt = 0
        opts = [path+"/0", path+"/1"]
        for pp in range(0, len(opts)):
            opt = opts[pp]
            count = 0
            for filename in os.listdir(opt):
                img_path = opt+"/"+filename
                image = Image.open(img_path)
                # bneck has a shape of [1, 1, 1, 2048]
                bneck = bGraph.get_bottle_neck_out(image)
                bneck = np.reshape(bneck, [-1, bneck.shape[-1]])
                _class = np.zeros((1, number_of_classes))
                _class[0][cls] = 1.0
                if cls_name == "nothing":
                    _pp = np.asarray([[0.0]])
                else:
                    if cls_name == "hdp":
                        _pp = np.asarray([[1.0]])
                    else:
                        _pp = np.asarray([[float(pp)]])
                _pp =np.reshape(_pp, [-1, 1])
                bneck_images.append((image, bneck, _class, _pp))
                total_samples += 1
                cnt += 1
                count +=1
            print("samples {} count {}".format(opt, count))
        print("class {} count {}".format(cls_name, cnt))
    data_dic["data"] = bneck_images
    with open(bottle_neck_pickel_path, 'wb') as handle:
        pickle.dump(data_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("processed data saved ", bottle_neck_pickel_path)
    print("total samples", total_samples)

    return data_dic["data"]

