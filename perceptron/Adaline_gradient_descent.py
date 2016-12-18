
'''
Compared to the previous one we will improve the convergence here with linear regression.
ie. we minimize the cost function j(w) and update w in a single shot
w' = -n*gardj(w)
also its weights are update for all date simultaneously
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
		print self.w_
		return self
	def net_input(self,X):
		return np.dot(X, self.w_[1:]) + self.w_[0]
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.01, 1, -1)

# This function plots and draws classification boundries
def plot_decision_regions(X, y, classifier, resolution=0.02):
	#setup marker generators and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	#plot decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:,1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx1.max())
	#plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)  

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
	#plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
	#plt.xlabel('Epochs')
	#plt.ylabel('# of missclassifications')
	#plt.show()
	# ---- plot decsion boundry on sample space
	#plot_decision_regions(X, y, ppn)
	#plt.xlabel('sepal length')
	#plt.ylabel('petal lenght')
	#plt.legend(loc='upper left')
	#plt.show()
	

if __name__ == "__main__":
	main()
