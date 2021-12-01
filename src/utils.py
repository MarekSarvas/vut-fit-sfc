from os import sendfile
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


def plot_cmeans(data, centers, membership, save_as=None):
    clusters_id = np.argmax(membership, axis=1)
    clusters_mem = np.max(membership, axis=1)
    data = data.T

    fig, ax = plt.subplots()
    scatter = ax.scatter(data[0], data[1], c=clusters_id, alpha=0.5,s=clusters_mem*100, label=clusters_id)
    print(centers)
    
    ax.scatter(centers.T[0], centers.T[1], c=np.arange(centers.shape[0]), marker='X', edgecolor='black', lw = 1)
    labels = []
    for i in range(centers.shape[0]):
        labels.append(f"Cluster {i}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


def plot_img(old_img, segmented):
    fig=plt.figure(figsize=(12,8),dpi=100)
    
    ax1=fig.add_subplot(1,2,1)
    ax1.imshow(old_img,cmap='gray')
    ax1.set_title('Greyscale image')

    ax2=fig.add_subplot(1,2,2)
    ax2.imshow(segmented)
    ax2.set_title('Segmentation with fuzzy Cmeans')
    
    plt.show()


if __name__=='__main__':
   plot_train_data(np.genfromtxt(sys.argv[1], delimiter=','), np.genfromtxt(sys.argv[2], delimiter=','))