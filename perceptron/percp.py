import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Perceptron(object):
	'''
	eta(float) : learning rate 0.0< eta < 1.0
	n_iter (int) : inc on training data set
	w_ (1d array) : weights after fitting
	errors_ (list) : number of missclassifiaction in every epoch
	'''
	def __init__(self, eta=0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter
	'''
	X [matrix] : [n_samples, n_features]
	y [1d aray] : [n_samples classes]
	'''
	def fit(self, X, y):
		self.w_ = np.zeros(1+ X.shape[1])
		self.errors_ = []
		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				#print "X: ", xi, "y: ", target, "y!: ", self.predict(xi)
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self
	def net_input(self,X):
		return np.dot(X, self.w_[1:]) + self.w_[0]
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.01, 1, -1)

def main():
	# ---- data process ----
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None) # load date set of iris from UCI
	df.tail() #view dataset
	y = df.iloc[0:100, 4].values #for 100 samples pick 4th column (ie. class)
	y = np.where(y == "Iris-setosa", -1, 1) #binary class Iris-setosa"(-1) Iris-versicolor(+1)
	X = df.iloc[0:100, [0, 2]].values # consider two features for 100 samples
	# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
	# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versocolor')
	# plt.xlabel('sepal length')
	# plt.ylable('petal length')
	# plt.legend(loc='upper left')
	# plt.show()
	
	# ---- training -----
	ppn = Perceptron(eta=0.1, n_iter=10)
	ppn.fit(X, y)
	# --- plot error
	plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('# of missclassifications')
	plt.show()
	

if __name__ == "__main__":
	main()
