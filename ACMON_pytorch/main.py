import gym
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from network import *
import numpy as np

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.encoder = encoder()
    self.decoder = decoder()

  def forward(self, x):
    x = Variable(torch.Tensor(x))
    x = self.encoder.encode(x)
    x = self.decoder.decode(x)
    return x


class game():
  def __init__(self):
    self.env = gym.make('MontezumaRevenge-v0')
    self.net = Net()
  def test(self):
    data_in = Variable(torch.randn(1, 1, 32, 32))
    out = self.net(data_in)
    print(out)
  def preprocess(self, img):
    img = img[20:]
    img = img[:160]
    img = np.reshape(img, [1, 3, 160, 160])
    img = img / 255.0
    return img
  def take_action(self, st, step):
    x = self.net(st)
    at = self.env.action_space.sample()
    st1, rt1, done, info = self.env.step(at)
    st1 = self.preprocess(st1)
    return st1, done
  def run(self):
    st = self.env.reset()
    st = self.preprocess(st)
    step = 1
    for _ in range(0, 10):
      st1, done = self.take_action(st, step)
      st = st1

def main():
  gm = game()
  gm.run()

if __name__ == "__main__":
  main()
