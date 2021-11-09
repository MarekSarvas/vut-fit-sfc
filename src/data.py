import numpy as np

class DataPoint:
    def __init__(self, x, y, clusters_num=3):
        self.x = x
        self.y = y
        self.clust_member = np.random.dirichlet(np.ones(clusters_num),size=1)
        self.curr_cluster = np.argmax(self.clust_member)
        self.curr_member_val = np.max(self.clust_member)

class Cluster:
    def __init__(self, c_id, x, y):
        self.id = c_id
        self.x = x
        self.y = y
        



if __name__=='__main__':
    d = DataPoint(1,1)
    print(d.clust_member, d.clust_member.shape, d.curr_cluster, d.curr_member_val)

