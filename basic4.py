# W.X+b operation feedforward neural network

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)

    return Z, cache
#Activation Unit output
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

#forward propagation operation for every layer
def forward_propagation(X, parameters, activation_back=args.activation):#activation may include sigmoid,relu,tanh ,softmax
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation=activation_back)
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],
                                          activation="softmax")#apply softmax for last layer
    caches.append(cache)

    return AL, caches
