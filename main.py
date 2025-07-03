import math

import numpy as np
import matplotlib.pyplot as plt

from node import Node
from net import Network

points = np.array([[0., 0., 1., 1.],
                   [0., 1., 0., 1.]])

xor_labels = np.array([0, 1, 1, 0])
or_labels = np.array([0, 1, 1, 1])

labels = xor_labels

def plot():
    label_colors = {0: 'r', 1: 'g'}
    colors = list(map(lambda x: label_colors[x], labels))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print(points[0, :])
    print(points[1, :])

    ax.scatter(points[0, :], points[1, :], c=list(colors), s=60)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


plot()
def runtrough():
    for x in range(100000):
        error_mid = 0
        for i in range(points.shape[1]):
        #Netz.midtrain(points,labels)
            pair = points[:, i]
            Netz.train(pair, labels[i])
            Netz.exponetialDecay(0.00023,0.0005,x,0.4)

            error_mid += abs(neuron2.preciseness)
        #Netz.coslineAnnealingRateScheduler(x,900,0.005,0.2)
            #if x == 150:
            #    abbilden()
        if x%100 == 0:
            Netz.abbilden_multicolor()
            print("---------------", neuron2.getLearningRate(),"-----predict_error---", neuron2.preciseness)
        if abs(error_mid/4) < 0.05:
            print("---------",x,"--------------")
            return x
            break
            #pass
            #Netz.train(pair, labels[i])
            #print(res)
    Netz.abbilden()
middle = 0
for i in range(10):
    neuron = Node(2, 4, 0.2, "sigmoid", "hidden")
    neuron2 = Node(4, 1, 0.2, "sigmoid", "output")
    neuron3 = Node(4, 1, 0.1, "sigmoid", "output")
    Netz = Network([neuron, neuron2])
    middle += runtrough()
middle = middle / 10
print(f"The final Average: {middle}")