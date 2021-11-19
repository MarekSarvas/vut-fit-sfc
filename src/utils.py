import matplotlib.pyplot as plt
import numpy as np
import sys

from numpy.lib.npyio import save


def plot_train_data(data, centroids):
    colors = (["blue", "red", "green", "yellow"])[:len(centroids)]
    plt.scatter(data.T[0], data.T[1], alpha=0.5)
    for cent, c in zip(centroids, colors):
        plt.scatter(cent.T[0], cent.T[1], color=c)
    

    plt.show()


def plot_cmeans(data, centers, clusters, save_as=None):
    clusters_id = np.argmax(clusters, axis=1)
    clusters_mem = np.max(clusters, axis=1)
    data = data.T
    cmap = 'tab10'
    fig, ax = plt.subplots()
    scatter = ax.scatter(data[0], data[1], c=clusters_id, alpha=0.5,s=clusters_mem*100, label=clusters_id)

    ax.scatter(centers.T[0], centers.T[1], c=np.arange(centers.shape[0]), marker='X', edgecolor='black', lw = 1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


if __name__=='__main__':
   plot_train_data(np.genfromtxt(sys.argv[1], delimiter=','), np.genfromtxt(sys.argv[2], delimiter=','))