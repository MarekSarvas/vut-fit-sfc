import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_train_data(data, centroids):
    colors = (["blue", "red", "green", "yellow"])[:len(centroids)]
    plt.scatter(data.T[0], data.T[1], alpha=0.5)
    for cent, c in zip(data, colors):
        plt.scatter(cent.T[0], cent.T[1], color=c)
    

    plt.show()


def plot_cmeans():
    
    
    pass


if __name__=='__main__':
   plot_train_data(np.genfromtxt(sys.argv[1], delimiter=','), np.genfromtxt(sys.argv[2], delimiter=','))