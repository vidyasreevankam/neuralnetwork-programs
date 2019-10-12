#Initialize the Parameters (W,b) weights and bias
def initialize_parameters(layer_dims):
    np.random.seed(1234)#keep random seed if you want to initialize some fixed weights
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * \
                                   np.sqrt(1 / layer_dims[l - 1])   #Hilbert Initialization (weights of dimension l*l-1)
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))#bias of dimension l*1

    return parameters



#layerdims={784,256,256,256,10}
