from matplotlib import image
import numpy as np
import argparse
from utils import plot_cmeans, plot_img
from PIL import Image

class FuzzyCMeans:
    def __init__(self, task, data, epochs=100, centr=3, q=2):
        self.q = q
        self.epochs = epochs
        self.clusters = centr
        self.data_points = data
        self.memberships = None

        if task == 'img':
            self._init_memberships(self.data_points.shape[0]) # membership of each data point to each cluster with shape (number of data points, number of clusters)
            self.run = self.run_clustering_img
        elif task == '2D':
            self._init_memberships(self.data_points.shape[0])
            self.run = self.run_clustering_2D
        else:
            print('Wrong task: {}'.format(task))
            exit()

        self.centroids = self._init_centroids()  # for plot
        
        #self.dist = np.zeros((self.data_points.shape[0], self.clusters))
        #self.update_w_dist(0)

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

    """
    def _load_data(self, path):
        return 

    def _load_centroids(self, path):
        return np.genfromtxt(path, delimiter=',')
    """

    # points are "randomly" assigned to clusters
    def _init_memberships(self, dim0):
        num_of_points = self.data_points.shape[0]
        data_index = np.arange(num_of_points)
        self.memberships = np.zeros((num_of_points, self.clusters))
        
        for i in range(self.clusters):
            idx = data_index%self.clusters == i
            self.memberships[idx, i] = 1

    #def _init_memberships1(self):
    #    self.memberships = np.random.dirichlet(np.ones(self.clusters),size=self.data_points.shape[0])

    #################### 2D data points #########################
    def _init_centroids(self):
        return np.zeros((self.clusters, 2))

    def update_centroids_2D(self):
        #print(self.data_points[0], self.memberships[0])

        
        #print(self.data_points[0] * self.memberships[0][0])
        #print(self.memberships.T[0][0])
        for i, c in enumerate(self.centroids):
            membership_tmp = self.memberships**self.q
            numerator_sum = np.sum(self.data_points * membership_tmp.T[i][:, np.newaxis], axis=0)
            denominator_sum = np.sum(membership_tmp.T[i])
            #print(denominator_sum, numerator_sum)
            self.centroids[i] = numerator_sum / denominator_sum

    def update_membership_2D(self):
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
                current_sum = np.sum(in_sum**(2/(self.q - 1))) # SUM (||x_i - c_j|| / ||x_i - c_k||)^(2/(q-1))
                #print(current_sum, np.power(in_sum, 2/(self.q - 1)))
                self.memberships[i][j] = 1 / current_sum

    ###############################################################################


    ################################# Image data ##################################
    def update_centroids_img(self):
        pass

    def update_membership_img(self):
        pass

    ###############################################################################
    def run_clustering_2D(self):
        plot_cmeans(self.data_points, self.centroids, self.memberships, save_as='old_mem.png')
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            self.update_centroids_2D()
            self.update_membership_2D()
            #self.update_w_dist(epoch)
        plot_cmeans(self.data_points, self.centroids, self.memberships, save_as='new_mem.png')
    
    def run_clustering_img(self):
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            self.update_centroids_img()
            self.update_membership_img()
        plot_img(None, None)

        
    """
    def update_centroids_img(self): 
        '''Compute weights'''
        c_mesh,idx_mesh = np.meshgrid(self.C,self.X)
        #print(c_mesh.shape, idx_mesh.shape)
        power = 2./(self.m-1)
        p1 = abs(idx_mesh-c_mesh)**power
        p2 = np.sum((1./abs(idx_mesh-c_mesh))**power,axis=1)
        #print(p1.shape, p2.shape)
        return 1./(p1*p2[:,None])

    """ 
    
def load_2D_data(path):
    try:
        data = np.genfromtxt(path, delimiter=',')
    except Exception as e:
        print('Wrong data path: {}'.format(e))
    return data

def load_img_data(path):
    try:
        img = Image.open(path)
        img = img.convert('L')  # greyscale
        img = np.array(img)
        data = img.flatten().astype('float')
    except Exception as e:
        print('Wrong data path: {}'.format(e))
    return data


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/marek/Documents/FIT/mit/sfc/vut-fit-sfc/data/data_9c_100_mu_15_var_6.csv', help='path to data csv')
    parser.add_argument('--clusters', default=9, type=int, help='path to centroids data csv')
    parser.add_argument('--ep', default=40, type=int, help='number of epochs for algorithm')
    args = parser.parse_args()

    img_path = '../img/mdb321.jpg'
    img_data = load_img_data(img_path)

    data_2D = load_2D_data(args.data)


    fcmeans = FuzzyCMeans('2D', data_2D, args.ep, args.clusters, q=3)
    fcmeans.run()