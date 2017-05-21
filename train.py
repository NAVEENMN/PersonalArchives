import pandas as pd
import numpy as np 
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from util import batch_generator
import argparse
import os
from keras.utils.np_utils import to_categorical
import keras
from sklearn.utils import shuffle

np.random.seed(0)
PATH = 'D:\\map_data\\'
PATH1 = 'D:\\map_data1\\'
PATHS = [PATH, PATH1]
IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNEL         = 1 
ACTIONS               = 4
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)

def load_data1(args):
    """
    Load training data and split it into training and validation set
    """
    X = list()
    y = list()
    for img in glob.glob(PATH+'*.png'):
        parts = img.split("\\")
        part = parts[2].split(".")
        screen= cv2.imread(img)
        part = part[0].split("_")
        caction = part[1]
        action = np.zeros(ACTIONS)  
        if caction == "1000":
            action = to_categorical(0, num_classes=ACTIONS)
        if caction == "0100":
            action = to_categorical(1, num_classes=ACTIONS)
        if caction == "0010":
            action = to_categorical(2, num_classes=ACTIONS)
        if caction == "0001":
            action = to_categorical(3, num_classes=ACTIONS)
        screen = cv2.resize(screen, (IMAGE_WIDTH, IMAGE_HEIGHT))
        screen = cv2.cvtColor( screen, cv2.COLOR_RGB2GRAY )
        screen = np.reshape(screen, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
        X.append(screen)
        y.append(action)
    #X, y = shuffle(X, y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    print(len(X_train))
    return X_train, X_valid, y_train, y_valid

def load_data(args):
    X = list()
    y = list()
    for path in PATHS:
        for img in glob.glob(PATH+'*.png'):
            parts = img.split("\\")
            part = parts[2].split(".")
            screen= cv2.imread(img)
            part = part[0].split("_")
            caction = part[1]
            action = np.zeros(ACTIONS)  
            if caction == "1000":
                action = to_categorical(0, num_classes=ACTIONS)
            if caction == "0100":
                action = to_categorical(1, num_classes=ACTIONS)
            if caction == "0010":
                action = to_categorical(2, num_classes=ACTIONS)
            if caction == "0001":
                action = to_categorical(3, num_classes=ACTIONS)
            X.append(img)
            y.append(action)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid

def build_model(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(ACTIONS))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=10000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    X_train, X_valid, y_train, y_valid = load_data(args)
    #build model
    model = build_model(args)
    #train model on data, it saves as model.h5 
    train_model(model, args,  X_train, X_valid, y_train, y_valid)


if __name__ == '__main__':
    main()

