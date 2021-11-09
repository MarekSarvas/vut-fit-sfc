import numpy as np
import csv
import argparse
import random

from utils import plot_train_data

# creates random data and cluster centroids and store them into csv files
def create_data(c_num, data, csv_name, mu, var):
    generated = None
    centroids_mu = None
    centroids_rand = None

    # loop through number of clusters to generate data
    for i in range(1, c_num+1):

        # generate datapoints [x,y] from normal distribution
        mu_x = random.randrange(-mu,mu)
        mu_y = random.randrange(-mu,mu)
        xs = np.random.normal(mu_x, random.uniform(1,var), data)
        ys = np.random.normal(mu_y, random.uniform(1,var), data)
        tmp = np.stack((xs, ys), axis=-1)
        
        # concatenate data from every cluster into one array
        if generated is None:
            generated = tmp
            centroids_mu = np.array([[mu_x, mu_y]])
            
            index = random.randrange(data)
            centroids_rand = np.array([[xs[index], ys[index]]])
        else:
            generated = np.concatenate((generated, tmp)) 
            centroids_mu = np.concatenate((centroids_mu, np.array([[mu_x, mu_y]])))
            
            index = random.randrange(data)
            centroids_rand = np.concatenate((centroids_rand, np.array([[xs[index], ys[index]]])))
        
    # store example data
    if csv_name is not None:
        print("Saving data into ../data/{}.csv".format(csv_name))
        np.savetxt('../data/{}.csv'.format(csv_name), generated, delimiter=",",newline='\n',fmt='%f')
        np.savetxt('../data/{}_centroids_mu.csv'.format(csv_name), centroids_mu, delimiter=",",newline='\n',fmt='%f')
        np.savetxt('../data/{}_centroids_rand.csv'.format(csv_name), centroids_rand, delimiter=",",newline='\n',fmt='%f')

    plot_train_data([generated, centroids_rand, centroids_mu])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', type=int, default=3)
    parser.add_argument('--data', type=int, default=100)
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--mu', type=int, default=5)
    parser.add_argument('--var', type=int, default=1.5)
    args = parser.parse_args()

    create_data(args.cluster_num, args.data, args.csv, args.mu, args.var)