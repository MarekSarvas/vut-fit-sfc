import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_train_data(data):
    colors = (["blue", "red", "green", "yellow"])[:len(data)]
    for d, c in zip(data, colors):
        plt.scatter(d.T[0], d.T[1], color=c )
    plt.show()

if __name__=='__main__':
   plot_train_data(np.genfromtxt(sys.argv[1], delimiter=','), np.genfromtxt(sys.argv[2], delimiter=','))