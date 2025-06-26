import numpy as np
import matplotlib.pyplot as plt

from node import Node

points = np.array([[0., 0., 1., 1.],
                   [0., 1., 0., 1.]])

xor_labels = np.array([-1, 1, 1, -1])
or_labels = np.array([-1, 1, 1, 1])

labels = or_labels

def plot():
    label_colors = {-1: 'r', 1: 'g'}
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

weights = np.random.rand(3)
eta = 0.2



#plot()
neuron = Node(2,1, 0.2, "sigmoid")
for i in range(points.shape[1]):
    pair = points[:, i]
    res = neuron.forward(pair)
    print(res)
