import numpy as np


class Network:
    def __init__(self, list):
        self.list = list

    def midtrain(self, X, y_ma):
        predictions_2d = [[] for _ in range(len(self.list)+1)]
        z_2d = [[] for _ in range(len(self.list))]
        for in_data in range(X.shape[1]):
            predictions_2d[0].append(X[:, in_data])
            for i in range(len(self.list)):
                predictions_2d[i+1].append(self.list[i].forward(predictions_2d[i][in_data]))
                z_2d[i].append(self.list[i].getZ())

            #predictions.append([data])
        y = np.stack(y_ma).mean(axis=0)
        predictions = []
        z = []
        for i in range(len(predictions_2d)):
            predictions.append(np.stack(predictions_2d[i]).mean(axis=0))
        for i in z_2d:
            z.append(np.stack(i).mean(axis=0))
        z.reverse()
        predictions.reverse()
        output_delta = []
        self.list.reverse()

        ## setting up teh averages

        ##
        self.list[0].setZ(z[0])
        output_delta.append(self.list[0].backward(predictions[1], y, predictions[0],0,0 ))
        for k in range(1, len(self.list)):
            self.list[k].setZ(z[k])
            output_delta.append(self.list[k].backward(predictions[k+1],y,predictions[k], output_delta[k-1], self.list[k-1].getMatrix() ))
        self.list.reverse()



    def anwenden(self, data):
        zw = data
        for i in range(len(self.list)):
            zw = self.list[i].forward(zw)
        return zw

    def train(self, data, y):
        predictions = [data]
        for i in range(len(self.list)):
            predictions.append(self.list[i].forward(predictions[i]))
        predictions.reverse()
        #predictions.append([data])
        output_delta = []
        self.list.reverse()
        output_delta.append(self.list[0].backward(predictions[1], y, predictions[0],0,0 ))
        for i in range(1, len(self.list)):
            output_delta.append(self.list[i].backward(predictions[i+1],y,predictions[i], output_delta[i-1], self.list[i-1].getMatrix() ))
        self.list.reverse()
        #hidden_1 = neuron.forward(data)
        #output = neuron2.forward(hidden_1)
        #output_delta = neuron2.backward(hidden_1, y, output, 0, 0)
        #hidden_1 = neuron.backward(data, y, hidden_1, output_delta, neuron2.getMatrix())