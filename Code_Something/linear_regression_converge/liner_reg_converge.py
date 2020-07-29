'''
modify gen_data.py with the liner equation you want
in form of y = m*x+c, randint will introduce light errors into it
and will be saved in data.txt
run linear_regression_converge.py which reads data makes prediction of the source queation by convergence. currently this code fails to converge if seed is very far.
'''
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from io import StringIO

threshold = 1

def f (x):
	val = (2 * x)  + 3
	val = val + rn.randint(-1, 1)
	return val

def pt(x_label, y_label, x_data, y_data, title):
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.grid(True)
	plt.scatter(x_data, y_data)
	plt.show()

content = []
x = []
y = []
with open("data.txt") as f:
	for line in f:
		line = line.rstrip('\n')
		ln = line.split(" ")
		x.append(float(ln[0]))
		y.append(float(ln[1]))

def get_error_for_hyp(th0, th1, x, y):
	error = 0
	b = np.array([[th0],[th1]]) # y = 2x+3
	data = list()
	for x1 in x:
		data.append([x1, 1])
	a = np.array(data)
	c = np.dot(a, b)
	h = c.tolist()
	error = 0
	for v in range(0, len(y)):
		e = y[v] - h[v][0]
		e = e * e
		error = error + e
	error = error / len(y)
	return error

def converge(seed, previous_error, flip, depth):
	depth = depth + 1
	if depth > 900:
		print "max depth"
		print seed, previous_error
		return seed
	net_error = get_error_for_hyp(seed, 2.5, x, y)
	if net_error <= threshold:
		print "found", seed
		exit()
		return seed
	else:
		if net_error > previous_error:
			print "net error > previous_error", net_error, previous_error 
			if flip == 0:
				flip = 1
				seed = seed - 0.01 
			else:
				flip = 0
				seed = seed + 0.01
		else:
			print "net error < previous_error", net_error, previous_error
			if flip == 0:
				seed = seed + 0.01
			else:
				seed = seed - 0.01
		converge(seed, net_error, flip, depth)
			
def main():
	th0 = converge(5, 1000, 1, 0)
	print th0
	#pt("x", "y", x, h, "line")

if __name__ == "__main__":
	main()
