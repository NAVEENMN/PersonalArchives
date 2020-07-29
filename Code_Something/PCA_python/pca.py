'''
3D to 2D
40 3-dimensional samples of two classes
'''
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.random.seed(234234782384239784)

def plot_data(class1_sample, class2_sample):
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	plt.rcParams['legend.fontsize'] = 10   
	ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
	ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

	plt.title('Samples for class 1 and class 2')
	ax.legend(loc='upper right')
	plt.show()

# step 1: arrange data set matrix
def gen_data():
	all_samples = None
	class1_sample, class2_sample = None, None
	# generate data for class 1
	mu_vec1 = np.array([0,0,0])
	cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
	class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
	assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"
	# generate data for class 2
	mu_vec2 = np.array([1,1,1])
	cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
	class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
	assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"
	# plot data
	# plot_data(class1_sample, class2_sample)
	all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
	return all_samples

# step 2: get mean vector
def get_mean_vector(all_samples):
	mean_x = np.mean(all_samples[0,:]) # all samples feature 1
	mean_y = np.mean(all_samples[1,:]) # all samples feature 2
	mean_z = np.mean(all_samples[2,:]) # all samples feature 3
	mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
	return mean_vector

# step 3: get co variance matrix
def get_covar_matrix(all_samples, mean_vector):
	covar_matrix = np.zeros((3,3))
	N = all_samples.shape[1]
	for i in range(N):
		covar_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)
	covar_matrix = covar_matrix/(N-1)
	return covar_matrix	

# step 4: get eigen values and vectors
def get_eigen(covar_matrix):
	eig_val_cov, eig_vec_cov = None, None
	eig_val_cov, eig_vec_cov = np.linalg.eig(covar_matrix)
	return eig_val_cov, eig_vec_cov

def main():
	all_samples = gen_data() # 3 X 40 matrix
	mean_vector = get_mean_vector(all_samples) # 1 X 3 matrix
	covar_matrix = get_covar_matrix(all_samples, mean_vector)
	eigen_values, eigen_vectors = get_eigen(covar_matrix)
	

if __name__ == "__main__":
	main()
