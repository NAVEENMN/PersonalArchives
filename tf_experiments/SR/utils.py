from defs import *
import numpy as np
from collections import deque
import random
#import cv2

def pre_process(state):
    state = np.asarray(state)
    #state = cv2.resize(state, (0, 0), fx=RESIZE, fy=RESIZE)
    #state = (state / 255.0)
    #latent = net.get_state_latent(state)
    state = (state / 127.5) - 1.0
    return state

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        # The right side of the deque contains the most recent experiences
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, tr):
        #[s, a, r, s2, done] = tr
        experience = tr #(s, a, r, s2, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def extend(self, temp_mem):
        while temp_mem.size() > 0:
            tr = temp_mem.buffer.popleft()
            temp_mem.count -= 1
            self.add(tr)

    def flush(self):
        self.buffer.clear()

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        s_batch = np.reshape([_[0] for _ in batch], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
        a_batch = np.reshape([_[1] for _ in batch], [-1, ACTION_SPACE])
        r_batch = np.reshape([_[2] for _ in batch], [-1, 1])
        s2_batch = np.reshape([_[3] for _ in batch], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
        done_batch = np.reshape([_[4] for _ in batch], [-1, 1])

        return s_batch, a_batch, r_batch, s2_batch, done_batch
