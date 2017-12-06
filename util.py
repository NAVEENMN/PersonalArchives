import numpy as np

def preprocess(img):
  img = img[20:]
  img = img[:160]
  img = np.reshape(img, [1, 160, 160, 3])
  img = img/255.0#(img/127.5)-1.0
  return img
