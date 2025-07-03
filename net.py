import math
import matplotlib.pyplot as plt
import numpy as np


class Network:
    def __init__(self, list):
        self.list = list

    def reset(self):
        for layer in self.list:
            layer.reset()

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
    def linearDescent(self, descend, stop):
        for i in self.list:
            zw = i.getLearningRate()
            if zw > stop:
                i.setLearningRate(zw-descend)
    # the smaller the more flat it is
    def exponetialDecay(self, decayRate, stop, run, learningRate):
        for i in self.list:
            zw = i.getLearningRate()
            if zw > stop:
                i.setLearningRate(learningRate*math.exp(-decayRate * run))
    def polynominalDecay(self, power, epoch, max_epoch, stop, learningRate):
        for i in self.list:
            zw = i.getLearningRate()
            if zw > stop:
                newRate = learningRate * (1-math.pow(epoch/max_epoch, power))
                i.setLearningRate(newRate)

    def coslineAnnealingRateScheduler(self, epoch, max_epoch, stop, learningRate):
        for i in self.list:
            newRate = stop + ((learningRate-stop)/2)*(1+math.cos(math.pi*epoch/max_epoch))
            i.setLearningRate(newRate)


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

    def abbilden(self):
        x = -0.5
        point_small  = np.empty((2,0))
        solution = np.empty((1,0))
        while x <= 1.5:
            y = -0.5
            while y <= 1.5:
                point = np.array([[x], [y]])
                point_small = np.hstack((point_small, point))
                if self.anwenden(point.T) < 0.5:
                    solution = np.hstack((solution, np.array([[0]])))
                else:
                    solution = np.hstack((solution, np.array([[1]])))
                y += 0.1
            x += 0.1

        label_colors = {0: 'r', 1: 'g'}
        colors = list(map(lambda x: label_colors[int(x)], solution.flatten()))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(point_small[0, :], point_small[1, :], c=list(colors), s=60)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        fig.tight_layout()
        plt.show()

    def abbilden_multicolor(self, border_red = 0.2, border_green = 0.2):
        x = -0.5
        point_small  = np.empty((2,0))
        solution = np.empty((1,0))
        while x <= 1.5:
            y = -0.5
            while y <= 1.5:
                point = np.array([[x], [y]])
                point_small = np.hstack((point_small, point))
                k = self.anwenden(point.T)
                if k < 0.2:
                    solution = np.hstack((solution, np.array([[0]])))
                elif k < 0.8:
                    solution = np.hstack((solution, np.array([[2]])))
                else:
                    solution = np.hstack((solution, np.array([[1]])))
                y += 0.1
            x += 0.1

        label_colors = {0: 'r', 1: 'g', 2: 'y'}
        colors = list(map(lambda x: label_colors[int(x)], solution.flatten()))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(point_small[0, :], point_small[1, :], c=list(colors), s=60)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        fig.tight_layout()
        plt.show()