import numpy as np
import argparse
from utils import plot_cmeans
import time

class FuzzyCMeans:
    def __init__(self, epochs=100, data_path=None, centr=3, q=2):
        self.q = q
        self.epochs = epochs
        self.clusters = centr
        self.data_points = None

        # load data
        if data_path is not None:
            try:
                self.data_points = self._load_data(data_path)
            except Exception as e:
                print('Wrong data file: {}'.format(e))

        self.memberships = self._init_memberships() # membership of each data point to each cluster with shape (number of data points, number of clusters)
        self.centroids = self._init_centroids()
        self.dist = np.zeros((self.data_points.shape[0], self.clusters))
        self.update_w_dist(0)

    def _load_data(self, path):
        return np.genfromtxt(path, delimiter=',')

    def _load_centroids(self, path):
        return np.genfromtxt(path, delimiter=',')
    
    def _init_memberships(self):
        return np.random.dirichlet(np.ones(self.clusters),size=self.data_points.shape[0])

    def _init_centroids(self):
        return np.zeros((self.clusters, 2))

    def update_w_dist(self, iter):

        for c in range(self.clusters):
            dist = np.linalg.norm(self.data_points - self.centroids[c], axis=1)
            dist = np.power(dist, 2)
            #print(dist[:10])
            #print(self.memberships[:10, c])
            self.dist[:, c] = dist
            #print(self.weighted_dist[:10])
        
        #print(np.sum(self.weighted_dist, axis=0))
        #print(self.memberships[0], self.memberships[:10, 0])
        #print(self.dist)
    def update_centroids(self):
        #print(self.data_points[0], self.memberships[0])

        
        #print(self.data_points[0] * self.memberships[0][0])
        #print(self.memberships.T[0][0])
        for i, c in enumerate(self.centroids):
            numerator_sum = np.sum(self.data_points * self.memberships.T[i][:, np.newaxis], axis=0)
            denominator_sum = np.sum(self.memberships.T[i])
            #print(denominator_sum, numerator_sum)
            self.centroids[i] = numerator_sum / denominator_sum

 
    def update_membership(self):
        #print(self.memberships[0])
        for j in range(self.clusters):
            for i, x in enumerate(self.data_points):
                #print('Cluster: {}, data: {}'.format(j+1, i))
                
                cloned_data = np.tile(x, (self.clusters,1))
                #print(cloned_data)
                #print(self.centroids)

                curr_centr_dist = np.linalg.norm(cloned_data - self.centroids[j], axis=1)  # || x_i - c_j ||
                all_distances = np.linalg.norm(cloned_data - self.centroids, axis=1)       # || x_i - c_k ||
                #print(curr_centr_dist)
                #print(all_distances)
                
                in_sum = curr_centr_dist / all_distances  # ||x_i - c_j|| / ||x_i - c_k||
                #print(in_sum)
                current_sum = np.sum(np.power(in_sum, 2/(self.q - 1))) # SUM (||x_i - c_j|| / ||x_i - c_k||)^(2/(q-1))
                #print(current_sum, np.power(in_sum, 2/(self.q - 1)))
                self.memberships[i][j] = 1 / current_sum


    def run_clustering(self):
        plot_cmeans(self.data_points, self.centroids, self.memberships, save_as='old_mem.png')
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            self.update_centroids()
            self.update_membership()
            self.update_w_dist(epoch)
            

        plot_cmeans(self.data_points, self.centroids, self.memberships, save_as='new_mem.png')
        
            


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/data_3c_100.csv', help='path to data csv')
    parser.add_argument('--clusters', default=3, type=int, help='path to centroids data csv')
    parser.add_argument('--ep', default=100, type=int, help='number of epochs for algorithm')

    args = parser.parse_args()

    fcmeans = FuzzyCMeans(args.ep, args.data, args.clusters, q=2)
    fcmeans.run_clustering()