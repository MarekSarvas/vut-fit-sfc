from matplotlib import image
import numpy as np
import argparse
from utils import plot_cmeans, plot_img
from PIL import Image

class FuzzyCMeans:
    def __init__(self, task, data_path, epochs=100, centr=3, q=2, args=None):
        self.q = q
        self.epochs = epochs
        self.clusters = centr
        #self.data_points = data
        self.memberships = None
        self.cmeans_args = args
        self.task = task

        if task == 'img':
            self.data_points = self.load_img_data(data_path)
            self._init_memberships(self.data_points.shape[0]) # membership of each data point to each cluster with shape (number of data points, number of clusters)
            self.run = self.run_clustering_img
            self.centroids = None 
            self.one_step = self.one_step_img
        elif task == 'points':
            self.data_points = self.load_2D_data(data_path)
            self._init_memberships(self.data_points.shape[0])
            self.run = self.run_clustering_2D
            self.centroids = self._init_centroids()  # for plot 
            self.one_step = self.one_step_2D
        else:
            print('Wrong task: {}'.format(task))
            exit()
        self.old_data_points = self.data_points
    ########################## UTILS ###############################
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


    def load_2D_data(self, path):
        try:
            data = np.genfromtxt(path, delimiter=',')
        except Exception as e:
            print('Wrong data path: {}'.format(e))
        return data

    def load_img_data(self, path):
        try:
            img = Image.open(path)
            img = img.resize((1024,1024),Image.ANTIALIAS)
            img = img.convert('L')  # greyscale
            self.grey_img = np.array(img)
            self.img_shape = self.grey_img.shape
            return self.grey_img.flatten().astype('float')
        except Exception as e:
            print('Wrong data path: {}'.format(e))
   
    # points are "randomly" assigned to clusters
    def _init_memberships(self, dim0):
        data_index = np.arange(dim0)
        self.memberships = np.zeros((dim0, self.clusters))
        
        for i in range(self.clusters):
            idx = data_index%self.clusters == i
            self.memberships[idx, i] = 1

    def reset(self):
        self.data_points = self.old_data_points
        self._init_memberships(self.data_points.shape[0])
        if self.task == "points":
            self.centroids = self._init_centroids()  # for plot 

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
        self.centroids = np.dot(self.data_points, self.memberships**self.q)/np.sum(self.memberships**self.q, axis=0)

    def update_membership_img(self):
        cent_m, data_m = np.meshgrid(self.centroids, self.data_points)
        data_from_centroids = abs(data_m - cent_m)**(2/(self.q - 1))
        all_distance_sum = np.sum(1/(abs(data_m - cent_m)**(2/(self.q - 1))), axis=1)
        self.memberships = 1/(data_from_centroids*all_distance_sum[:, None])
    
    def reconstruct_img(self):
        recon = np.argmax(self.memberships, axis=1)
        return recon.reshape(self.img_shape)

    ###############################################################################
    def run_clustering_2D(self):
        #plot_cmeans(self.data_points, self.centroids, self.memberships, save_as=args.save_img+"_old")
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            self.update_centroids_2D()
            self.update_membership_2D()
            # self.update_w_dist(epoch)
        #plot_cmeans(self.data_points, self.centroids, self.memberships, save_as=args.save_img)
    
    def run_clustering_img(self):
        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            self.update_centroids_img()
            self.update_membership_img()
        #plot_img(self.grey_img, self.reconstruct_img(), args.save_img)

    def one_step_2D(self):
        self.update_centroids_2D()
        self.update_membership_2D()

    def one_step_img(self):
        self.update_centroids_img()
        self.update_membership_img()
        



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/marek/Documents/FIT/mit/sfc/vut-fit-sfc/data/data_9c_100_mu_15_var_6.csv', help='path to data csv')
    parser.add_argument('--num_clusters', default=9, type=int, help='path to centroids data csv')
    parser.add_argument('--epochs', default=40, type=int, help='number of epochs for algorithm')
    parser.add_argument('--q', default=2, type=int, help='fuzziness')
    parser.add_argument('--task', default="points", type=str, help='cmeans on 2D data points or img')
    parser.add_argument('--save_img', default="cmeans_plot", type=str, help='name of file to save cmeans plot')
    args = parser.parse_args()

    #img_path = '../img/mdb321.jpg'
    #data_path = img_path
    #data_path = data_2D

    fcmeans = FuzzyCMeans(args.task, args.data_path, args.epochs, args.num_clusters, args.q, args)
    fcmeans.run()