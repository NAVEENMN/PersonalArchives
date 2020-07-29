import numpy as np
import random as rn
def f(x):
	val = (2 * x)  + 3
	val = val + rn.randint(-1, 1)
	return val

def main():
	x = np.arange(0.0, 200.0, 0.1)
	y = list()
	fl = open('data.txt','w')
	for x1 in x:
		y.append(f(x1))
		dat = str(x1) + " " + str(f(x1))+"\n" 
		fl.write(dat)
	fl.close() 

if __name__ == "__main__":
	main()
