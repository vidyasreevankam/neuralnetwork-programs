#optimization algorithms
#Update using Gradient Descent
def gd_update(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

#initialization for Momentum
def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.array(np.zeros(shape=parameters["W" + str(l + 1)].shape))
        v["db" + str(l + 1)] = np.array(np.zeros(shape=parameters["b" + str(l + 1)].shape))

    return v

#Updation for Momentum
def momentum_update(parameters, grads, m, gamma, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        m["dW" + str(l + 1)] = gamma * m["dW" + str(l + 1)] + learning_rate * grads["dW" + str(l + 1)]
        m["db" + str(l + 1)] = gamma * m["db" + str(l + 1)] + learning_rate * grads["db" + str(l + 1)]

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - m["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - m["db" + str(l + 1)]

    return parameters, m

#Updation for Nestrov Accelarated Momentum
def nag_update(parameters, grads, m, gamma, learning_rate, AL, Y_batch, caches, activation_back):
    L = len(parameters) // 2
    parameters_PV = copy.deepcopy(parameters)

    for l in range(L):

        m["dW" + str(l + 1)] = gamma * m["dW" + str(l + 1)]
        m["db" + str(l + 1)] = gamma * m["db" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - m["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - m["db" + str(l + 1)]

        grads = backward_propagation(AL, Y_batch, caches, activation_back)

        for t in range(L):
            m["dW" + str(t + 1)] = gamma * m["dW" + str(t + 1)] + learning_rate * grads["dW" + str(t + 1)]
            m["db" + str(t + 1)] = gamma * m["db" + str(t + 1)] + learning_rate * grads["db" + str(t + 1)]
            parameters["W" + str(t + 1)] = parameters_PV["W" + str(t + 1)] - m["dW" + str(t + 1)]
            parameters["b" + str(t + 1)] = parameters_PV["b" + str(t + 1)] - m["db" + str(t + 1)]
            parameters_PV["W" + str(t + 1)] = copy.deepcopy(parameters["W" + str(t + 1)])
            parameters_PV["W" + str(t + 1)] = copy.deepcopy(parameters["W" + str(t + 1)])
    return parameters, m

#Initialization for Adam
def initialize_adam(parameters):
    L = len(parameters) // 2
    m = {}
    v = {}

    for l in range(L):
        m["dW" + str(l + 1)] = np.zeros(shape=parameters["W" + str(l + 1)].shape)
        m["db" + str(l + 1)] = np.zeros(shape=parameters["b" + str(l + 1)].shape)
        v["dW" + str(l + 1)] = np.zeros(shape=parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(shape=parameters["b" + str(l + 1)].shape)

    return m, v

#Updation for Adam
def adam_update(parameters, grads, m, v, t, learning_rate=args.lr,
                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    m_corrected = {}
    v_corrected = {}

    for l in range(L):
        m["dW" + str(l + 1)] = beta1 * m["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        m["db" + str(l + 1)] = beta1 * m["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        m_corrected["dW" + str(l + 1)] = m["dW" + str(l + 1)] / (1 - beta1 ** t)
        m_corrected["db" + str(l + 1)] = m["db" + str(l + 1)] / (1 - beta1 ** t)

        v["dW" + str(l + 1)] = beta2 * v["dW" + str(l + 1)] + (1 - beta2) * np.power(grads["dW" + str(l + 1)], 2)
        v["db" + str(l + 1)] = beta2 * v["db" + str(l + 1)] + (1 - beta2) * np.power(grads["db" + str(l + 1)], 2)

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta2 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta2 ** t)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * m_corrected[
            "dW" + str(l + 1)] / (np.sqrt(v_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * m_corrected[
            "db" + str(l + 1)] / (np.sqrt(v_corrected["db" + str(l + 1)] + epsilon))

    return parameters, m, v

