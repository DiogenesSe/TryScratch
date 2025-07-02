import numpy as np
import random

class Node:
    def __init__(self, X_input_size, data_output_size, eta, activation, position):
        self.data_input_size = X_input_size
        self.data_output_size = data_output_size
        self.eta = eta
        # n x p has to be machted to p x y

        #self.matrix = np.random.rand(X_input_size, data_output_size)
        # testing of negative weights aswell
        self.matrix = np.random.uniform(-0.5, 0.5, (X_input_size, data_output_size))
        self.position = position
        #to avoid dying rlu 0.01 else 0.0 would be acceptable
        #self.bias = np.full((1,data_output_size),0.1)
        #test with other
        self.bias = np.zeros((1, data_output_size))
        # dictionary for all the activaion function
        if isinstance(activation, str):
            dic= {'sigmoid': (self.sigmoid, self.sigmoid_deriv), 'relu': (self.relu, self.relu_deriv), 'sign': (self.sign, self.sign_deriv), 'tanh':(self.tanh, self.tanh_deriv) }
            if activation not in dic:
                raise ValueError(f"Unknown activation: {activation}")
            self.activation, self.activation_deriv = dic[activation]

        else:
            self.activation = activation

    #Error Depending on position
    def error(self, y_true, y_pred, output_delta, weights_hidden_output):
        if self.position == "output":
            return  self.meanSquareError(y_true, y_pred)
        elif self.position == "hidden":
            return np.dot(output_delta, weights_hidden_output.T)
        else:
            return 0

    #tanh output x-> -inf : -1, x -> inf: 1
    #o(x) = (e^z - e ^ -z) / (e ^ z + e ^ -z)
    def tanh(self, x):
        return np.tanh(x)
    def tanh_deriv(self, x):
        return 1- np.tanh(x)**2
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
    def sign_deriv(self, x):
        return 0

    def standartLoss(self,  y_true, y_pred):
        return y_true -y_pred
    #Good for classification problems. CE = -(y * log(ŷ)+(1-y)*log(1-ŷ))
    # y = real label, ŷ = prediction
    def crossEntropyLoss(self, y_true, y_pred,eps=1e-15 ):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    #Good for regression problems. For this f(x) = 1 E n : (x-y)^ 2
    def meanSquareError(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    # Bad grows with data set size
    def sumSquaredError(self, y_true,y_pred):
        return np.sum((y_true - y_pred)**2)

    #data x wheight_matrix + biase = z , activation(z) = prediction
    def forward(self, data):
        data = data.reshape(1, -1)
        self.z = np.dot(data, self.matrix) + self.bias
        prediction = self.activation(self.z)
        return prediction

    #gradient weight = learning rate * error term *
    def backward(self, output_prev, y, y_predict, output_delta, weights_previous):
        #error if output = y_predict -y else hidden = w_danach * y_predict
        error = self.error(y, y_predict, output_delta, weights_previous)
        #delta = error *f'(y_predict)
        delta = error *  self.activation_deriv(self.z)
        #wheight = weight+ ( output_prev * error* f'(y_predict) * eta)
        self.matrix += np.dot(output_prev.reshape(-1, 1) ,delta) * self.eta
        #bias = error * f'(y_predict) * eta
        self.bias += np.sum(delta, axis=0, keepdims=True) * self.eta
        return delta
    def getZ(self):
        return self.z
    def setZ(self, z):
        self.z = z
    def getMatrix(self):
        return self.matrix
    def getBias(self):
        return self.bias

    def setBias(self, bias):
        self.bias = bias
    def setMatrix(self, matrix):
        self.matrix = matrix
    def returnAll(self):
        print("---------- MATRIX ----------")
        print(self.matrix)
        print("----------- BIAS -----------")
        print(self.bias)

