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


def abbilden():
    x = -0.5
    point_small  = np.empty((2,0))
    solution = np.empty((1,0))
    while x <= 1.5:
        y = -0.5
        while y <= 1.5:
            point = np.array([[x], [y]])
            point_small = np.hstack((point_small, point))
            if anwenden(point.T) < 0.5:
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


neuron = Node(2,4, 0.01, "sigmoid", "hidden")
neuron2 = Node(4,1,0.01, "sigmoid", "output")
neuron3 = Node(4,1,0.1, "sigmoid", "output")
Netz = Network([neuron, neuron2])
def train(data,y):
    hidden_1 = neuron.forward(data)
    output = neuron2.forward(hidden_1)
    output_delta = neuron2.backward(hidden_1,y,output,0,0)
    hidden_1 = neuron.backward(data,y,hidden_1, output_delta,neuron2.getMatrix() )

def anwenden(data):
    return Netz.anwenden(data)



plot()
for x in range(100000):
    for i in range(points.shape[1]):
    #Netz.midtrain(points,labels)
        pair = points[:, i]
        Netz.train(pair, labels[i])
        #if x == 150:
        #    abbilden()
    if x%100 == 0:
        abbilden()
        #pass
        #Netz.train(pair, labels[i])
        #print(res)
abbilden()