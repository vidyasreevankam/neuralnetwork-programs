import numpy as np
#parameter initialization
layer_dims=(n_inputs,n_hidden,n_hidden,n_output)
def initialize_parameters(layer_dims):
	np.random.seed(1234)
	parameters = {}
	L=len(layer_dims)
	for l in range(1, L):
  		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
  		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
  		return parameters
net=initialize_parameters(34,2,2,3)
for layer in net:
	print(layer)
