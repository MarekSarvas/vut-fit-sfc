import numpy as np

class FuzzyCMeans:
    def __init__(self, data_path, centr_path, num_clusters, q=2):
        self.data_points = self._load_data(data_path)
        self.centroids = self._load_centroids(centr_path)
        
        self.clusters = num_clusters
        self.q = q
        self.membership = self.update_membership()

    def _load_data(self, path):
        return np.genfromtxt(path, delimiter=',')

    def _load_centroids(self, path):
        return np.genfromtxt(path, delimiter=',')

    def update_centroids(self):
        pass
    
    def update_membership(self):
        pass






if __name__=='__main__':
    pass
