#Different ACtivation Functions  and their Differentiation
def softmax(Z):
    e_x = np.exp(Z - np.max(Z))
    A = e_x / e_x.sum(axis=0)
    cache = Z
    return A, cache


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tanh_backward(dA, cache):
    Z = cache
    dZ = 1 - np.square(np.tanh(Z))
    assert (dZ.shape == Z.shape)
    return dZ
