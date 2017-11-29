import numpy as np

def rgb_to_gray(img):
  grayImage = np.zeros(img.shape)
  R = np.array(img[:, :, 0])
  G = np.array(img[:, :, 1])
  B = np.array(img[:, :, 2])
  R = (R *.299)
  G = (G *.587)
  B = (B *.114)
  Avg = (R+G+B)
  grayImage = img

  for i in range(3):
    grayImage[:,:,i] = Avg

  return grayImage

def preprocess(img):
  #img = rgb_to_gray(img)
  img = img[20:]
  img = img[:160]
  img = np.reshape(img, [1, 160, 160, 3])
  return img
