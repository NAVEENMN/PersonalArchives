import os
import cv2
import glob
import json
import keras
import utils
import argparse
import numpy as np
import pandas as pd
import utils.game_manager as gm
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

Train          = True

model          = None
IMAGE_WIDTH    = 320
IMAGE_HEIGHT   = 240
IMAGE_CHANNELS = 1
ACTIONS        = 4
INPUT_SHAPE    = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
CSV_PATH       = 'replays\\'
IMAGES_PATH    = 'D:\\map_data\\'

np.random.seed(0)

def setup():
    with open('setup.json') as data_file:
        data = json.load(data_file)
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default=data["images_path"])
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=10000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=bool,  default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()
    return args

def load_data(args):
    X = list()
    y = list()
    for replay in glob.glob(CSV_PATH+'*.csv'):
        df = pd.read_csv(replay, sep=',',header=None)
        print(replay)
        dir_path = replay.split("\\")[1]
        dir_path = dir_path.split(".")[0]
        dir_path = dir_path+"\\"
        screens = df[1][0]
        actions = df[1][1]
        screens = pd.DataFrame(screens.split(','))
        actions = pd.DataFrame(actions.split(','))
        for x in range(0, len(screens)):
            img = dir_path+screens[0][x]
            print(IMAGES_PATH+img)
            if os.path.isfile(IMAGES_PATH+img): 
                action = to_categorical(actions[0][x], num_classes=ACTIONS)  
                X.append(img)
                y.append(action)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    print(len(X_train))
    print(len(y_train))
    return X_train, X_valid, y_train, y_valid


"""
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
"""
def build_model(args):
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
    model.fit_generator(gm.batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=gm.batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

def fly_drone(args):
    model = load_model(args.model)
    for x in range(0, 5):
        screen = gm.grab_screen(region=(0, 40 , 1920, 1080))
        screen = cv2.resize(screen, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
        screen = cv2.cvtColor( screen, cv2.COLOR_RGB2GRAY )
        screen = np.reshape(screen, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
        print(screen.shape)
        action = model.predict(screen, batch_size=1)
        print(action[0])
        take_action = [0, 0, 0, 0]
        ind = np.argmax(action[0])
        take_action[ind] = 1
        #gm.Actions(take_action)
        data = dict()
        data['action_index'] = 1;
        result = firebase.put('/actions', "action", data)
        print(result)

def train():
    args = setup()

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

def test():
    parser = argparse.ArgumentParser(description='Autonomous Drone')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()
    fly_drone(args)

def main():
    if Train:
        train()
    else:
        test()

if __name__ == '__main__':
    main()

