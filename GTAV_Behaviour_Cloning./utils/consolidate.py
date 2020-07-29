import glob, os
import pickle

DIR_PATH = 'D:\\map_data2\\'

def main():
	dirs = os.listdir(DIR_PATH)
	DATA = dict()
	for file in dirs:
		print(file)
		with open(DIR_PATH+file, "rb") as pkl:
			screens = pickle.load(pkl)
			for key in screens.keys():
				DATA[key] = screens[key]
	with open(DIR_PATH+"new", "wb") as pkl:
			pickle.dump(DATA, pkl, protocol=pickle.HIGHEST_PROTOCOL)
	print (DATA.keys())


if __name__ == "__main__":
	main()