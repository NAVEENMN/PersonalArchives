from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer

'''
lets train network for two actions
forward and backword
1 0
'''
def build_data():
	ds = SupervisedDataSet(2, 1)
	for x in range(0, 10):
		ds.addSample((0, 0), (0,))
		ds.addSample((0, 1), (1,))
		ds.addSample((1, 0), (1,))
		ds.addSample((1, 1), (0,))
	return ds

def get_net():
	ds = build_data()
	net = buildNetwork(2, 3, 1, bias=True, hiddenclass=SigmoidLayer)
	trainer = BackpropTrainer(net, ds)
	trainer.trainUntilConvergence()
	return net

def main():
	net = get_net()
	test = [[0,0],[0,1],[1,0],[1,1]]
	for sample in test:
		out = net.activate(sample)
		print sample, out

if __name__ == "__main__":
	main()
