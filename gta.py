import os
import cv2
import sys
import time
import keras
import utils
import random
import pickle
import signal
import argparse
import numpy as np
from collections import deque
import utils.game_manager as gm
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

SCREEN_SHOTS_PATH = 'D:\\GTA_BC\\'
MEMORY_PATH       = "memory\\"
TEST_PATH         = "test\\"
IMAGE_WIDTH       = 320
IMAGE_HEIGHT      = 240
IMAGE_CHANNELS    = 3
model             = None
ACTIONS           = 4
INPUT_SHAPE       = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

REPLAY_MEMORY     = 50000

def signal_handler(signal, frame):
		print('You pressed Ctrl+C!')
		pkl.close()
		sys.exit(0)

def visulize_a_image(image, ai_action, hm_action, mode):
	if mode == "test":
		image = np.reshape(image, (1011, 1811, 3))
	else:
		image = np.reshape(image, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
	cv2.putText(image,"AI: "+str(ai_action), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
	cv2.putText(image,"HM: "+str(hm_action), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
	cv2.imshow("recap", image)
	cv2.imwrite("memory\\"+time.strftime("%Y%m%d-%H%M%S")+".png",image)
	cv2.waitKey(1)

def setup():
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default=MEMORY_PATH)
	parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
	parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
	parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=30)
	parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=10000)
	parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
	parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=bool,  default='true')
	parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
	args = parser.parse_args()
	return args


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
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(ACTIONS))
	model.summary()

	return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
	checkpoint = ModelCheckpoint('models\\model-{epoch:03d}.h5',
								 monitor='val_loss',
								 verbose=1,
								 save_best_only=args.save_best_only,
								 mode='auto')
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
	tensorboard = TensorBoard(log_dir='Graph\\', histogram_freq=0,
                          write_graph=True, write_images=False)
	model.fit_generator(gm.batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
						args.samples_per_epoch,
						args.nb_epoch,
						max_q_size=1,
						validation_data=gm.batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
						nb_val_samples=len(X_valid),
						callbacks=[checkpoint, tensorboard],
						verbose=1)

def data_collection(PATH, mode):
	time.sleep(5)
	timestr = time.strftime("%Y%m%d-%H%M%S")
	print("game-"+timestr)
	path = PATH+timestr+"\\"
	game = list()
	quit = False
	count = 0
	while not quit:
		gstr = time.strftime("%Y%m%d-%H%M%S")
		#fname = SCREEN_SHOTS_PATH+gstr
		count = count + 1
		screen, action, quit = gm.get_frame_action(mode)
		#cv2.imwrite(fname, screen)
		game.append([screen, action])
		if count >= 1000:
			with open(PATH+gstr, "wb") as pkl:
				pickle.dump(game, pkl, protocol=pickle.HIGHEST_PROTOCOL)
			pkl.close()
			count = 0
			del game[:]
			print("saved-"+"memory\\"+gstr)

def load_data(args):
	D = deque()
	memories = os.listdir(MEMORY_PATH)
	random.shuffle(memories)
	print(len(memories))
	for memory in memories:
		with open(MEMORY_PATH+memory, "rb") as pkl:
			game = pickle.load(pkl)
			for gpair in game:
				[screen, action] = gpair
				#visulize_a_image(screen, action, "collect")
				D.append([screen, action])
				if len(D) > REPLAY_MEMORY:
					#D.popleft()
					break
	#cv2.destroyAllWindows()
	random.shuffle(D)
	X = [d[0] for d in D]
	y = [d[1] for d in D]
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
	return X_train, X_valid, y_train, y_valid

def train(args):
	X_train, X_valid, y_train, y_valid = load_data(args)
	model = build_model(args)
	train_model(model, args, X_train, X_valid, y_train, y_valid)

# One GPU
def testgame():
	parser = argparse.ArgumentParser(description='Autonomous Drone')
	parser.add_argument('model',type=str,help='Path to model h5 file. Model should be on the same path.')
	args = parser.parse_args()
	model = load_model(args.model)
	memories = os.listdir(TEST_PATH)
	for memory in memories:
		with open(TEST_PATH+memory, "rb") as pkl:
			game = pickle.load(pkl)
			for gpair in game:
				[oscreen, hm_action] = gpair
				screen = gm.preprocess(oscreen)
				ai_action = model.predict(screen, batch_size=1)
				take_action =[0, 0, 0, 0]
				ind = np.argmax(ai_action[0])
				print(ai_action)
				take_action[ind] = 1
				visulize_a_image(oscreen, take_action, hm_action, "test")
	cv2.destroyAllWindows()

# more than one GPU
def test():
	parser = argparse.ArgumentParser(description='Autonomous Drone')
	parser.add_argument('model',type=str,help='Path to model h5 file. Model should be on the same path.')
	args = parser.parse_args()
	model = load_model(args.model)
	time.sleep(5)
	quit = False
	while not quit:
		screen, action, quit = gm.get_frame_action("test")
		print(screen)
		action = model.predict(screen, batch_size=1)
		print(action)
		take_action = np.zeros((1, ACTIONS))
		ind = np.argmax(action[0])
		take_action[ind] = 1
		gm.Actions(take_action)


def main():
	mode = 3 # 1 data collect, 2 train, 3 test
	if mode == 1:
		data_collection(MEMORY_PATH, "collect")
	if mode == 2:
		args= setup()
		train(args)
	if mode == 3:
		data_collection(TEST_PATH, "test")
		time.sleep(2)
		testgame()
		#test()

if __name__ == "__main__":
	main()