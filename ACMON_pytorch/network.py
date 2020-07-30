import torch.nn as nn
import torch.nn.functional as F

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL
BATCH_SHAPE = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]

# ------ Encoder ----- #
class encoder(nn.Module):
  def __init__(self):
    super(encoder, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 10)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.conv3 = nn.Conv2d(16, 20, 3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(16 * 35 * 35, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def encode(self, st):
    x = F.max_pool2d(F.relu(self.conv1(st)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
      size = x.size()[1:]  # all dimensions except the batch dimension
      num_features = 1
      for s in size:
          num_features *= s
      return num_features

# ------ Decoder ----- #
class decoder(nn.Module):
  def __init__(self):
    super(decoder, self).__init__()
    self.fc1 = nn.Linear(10, IMAGE_PIXELS*4)
    self.conv1 = nn.Conv2d(3, 64, 25)
    self.conv2 = nn.Conv2d(64, 32, 20)
    self.conv3 = nn.Conv2d(32, 30, 10)
    self.conv4 = nn.Conv2d(30, 3, 8)
    
  def decode(self, st_z):
    x = self.fc1(st_z)
    x = x.view([-1, IMAGE_CHANNEL, IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2])
    print(x.shape)
    x = F.relu(self.conv1(x))
    print(x.shape)
    x = F.relu(self.conv2(x))
    print(x.shape)
    x = F.relu(self.conv3(x))
    print(x.shape)
    x = F.relu(self.conv4(x))
    print(x.shape)
    print(" ")
    return x








