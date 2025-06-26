import numpy as np
import random

class Node:
    def __init__(self, X_input_size, data_output_size, eta, activation):
        self.data_input_size = X_input_size
        self.data_output_size = data_output_size
        self.eta = eta
        self.matrix = np.random.rand(X_input_size, data_output_size)
        #to avoid dying rlu 0.01 else 0.0 would be acceptable
        self.bias = np.full((data_output_size,1),0.01)
        # dictionary for all the activaion function
        if isinstance(activation, str):
            dic= {'sigmoid': self.sigmoid, 'relu': self.relu, 'sign': self.sign}
            if activation not in dic:
                raise ValueError(f"Unknown activation: {activation}")
            self.activation = dic[activation]
        else:
            self.activation = activation

    #tanh output x-> -inf : -1, x -> inf: 1
    #o(x) = (e^z - e ^ -z) / (e ^ z + e ^ -z)
    def tanh(self, x):
        return np.tanh(x)

    # sigmoid(x) = 1 / (1 + exp(-x))
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s*(1-s)

    # 0 and if anything, x > 0 then x
    def relu(self, x):
        return max(0, x)
    def relu_deriv(self, x):
        if x < 0:
            return 0
        else:
            return 1

    #Posible output x= 0 then 0 if x >0 then 1 ELSE -1
    def sign(self, x):
        return np.sign(x)

    #Good for classification problems. CE = -(y * log(ŷ)+(1-y)*log(1-ŷ))
    # y = real label, ŷ = prediction
    def crossEntropyLoss(self, y_pred, y_true,eps=1e-15 ):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    #Good for regression problems. For this f(x) = 1 E n : (x-y)^ 2
    def meanSquareError(self, X, Y):
        return np.mean((X - Y)**2)

    # Bad grows with data set size
    def sumSquaredError(self, X,Y):
        return np.sum((X - Y)**2)

    #data x wheight_matrix + biase = z , activation(z) = prediction
    def forward(self, data):
        z = np.dot(data, self.matrix) + self.bias
        prediction = self.activation(z)
        return prediction

    def backward(self, y ):

        pass



