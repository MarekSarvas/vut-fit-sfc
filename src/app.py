import sys
from PIL.Image import new
from PyQt5.QtWidgets import QDialog, QApplication, QGridLayout, QHBoxLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget, QLabel, QStackedLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib
import random
import argparse
import time
import numpy as np
from fuzzy_cmeans import FuzzyCMeans
matplotlib.use("Qt5agg")


class Window(QWidget):
    def __init__(self, args, parent=None):
        super(Window, self).__init__(parent)
        
        self.FCM = FuzzyCMeans("points", args.data_path, args.epochs, args.num_clusters, args.q, args)
        self.stop = False # pause 2D clustering
        self.epoch = 0

        self.FCM_img = FuzzyCMeans("img", args.data_path_img, args.epochs_img, args.num_clusters_img, args.q_img)
        self.active_img = 0 # 0 for greyscale 1 for segmented
        self.stop_img = False # pause segmentation loop 
        self.epoch_img = 0
        # figures for both tasks
        self.figure, self.ax = plt.subplots()
        self.figure.tight_layout()
       
        self.figure_img1, self.ax_img1 = plt.subplots()
        self.figure_img1.tight_layout()
        
        # canvas for plots
        self.canvas = FigureCanvas(self.figure)
        self.canvas_img1 = FigureCanvas(self.figure_img1)

        

        mainLayout = QGridLayout()
        vLayout1 = QVBoxLayout()

        # TAB 1.1
        # Buttons for points clustering
        self.button = QPushButton('Run')
        self.button.clicked.connect(self.run_clustering)
        self.button3 = QPushButton('Stop')
        self.button3.clicked.connect(self.stop_clustering)
        self.button2 = QPushButton('Reset')
        self.button2.clicked.connect(self.reset_clustering)
        
        self.tab1_1 = QWidget()
        self.tab1_1.layout = QVBoxLayout()
        self.plot_layout1 = QVBoxLayout()
        self.btn_layout1 = QHBoxLayout()
        self.tab1_1.layout.addLayout(self.plot_layout1)
        self.tab1_1.layout.addLayout(self.btn_layout1)
    
        self.plot_layout1.addWidget(self.canvas)
        self.btn_layout1.addWidget(self.button)
        self.btn_layout1.addWidget(self.button3)
        self.btn_layout1.addWidget(self.button2)
        self.tab1_1.setLayout(self.tab1_1.layout)

        # TAB 1.2
        # Buttons for image segmentation
        self.btn = QPushButton('Run segmentation')
        self.btn.clicked.connect(self.run_segmentation)
        self.btn1 = QPushButton('Grey/Segmented')
        self.btn1.clicked.connect(self.switch)
        self.btn2 = QPushButton('Stop')
        self.btn2.clicked.connect(self.stop_segmentation)
        self.btn3 = QPushButton('Reset')
        self.btn3.clicked.connect(self.reset_segmentation)
        
        self.tab1_2 = QWidget()
        self.tab1_2.layout = QVBoxLayout()
        self.plot_layout = QVBoxLayout()
        self.btn_layout = QHBoxLayout()
        self.tab1_2.layout.addLayout(self.plot_layout)
        self.tab1_2.layout.addLayout(self.btn_layout)
        self.plot_layout.addWidget(self.canvas_img1)
        
        self.btn_layout.addWidget(self.btn)
        self.btn_layout.addWidget(self.btn1)
        self.btn_layout.addWidget(self.btn2)
        self.btn_layout.addWidget(self.btn3)
        self.tab1_2.setLayout(self.tab1_2.layout)
        

        # set tabs in main app
        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab1_1, 'Clustering')
        self.tabs.addTab(self.tab1_2, 'Segmentation')
        mainLayout.addWidget(self.tabs, 0, 0)
        self.setLayout(mainLayout)
        cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)


        # init plots
        self.figure.canvas.draw_idle()
        self.figure.canvas.start_event_loop(0.001)
        self.plot_cmeans(self.FCM.data_points, self.FCM.centroids, self.FCM.memberships)
        
        self.figure_img1.canvas.draw_idle()
        self.plot_segmentation()


    def onclick(self, event):
        if event.ydata is not None and event.xdata is not None:
            print("Added point at [{}, {}]".format(event.xdata, event.ydata))
            
            new_data = np.asarray([[event.xdata, event.ydata]])
            new_mem = np.zeros(self.FCM.clusters)
            new_mem = new_mem[None, :]
            new_mem[0,0] = 1
       
            self.FCM.data_points = np.concatenate((self.FCM.data_points, new_data))
            self.FCM.memberships = np.concatenate((self.FCM.memberships, new_mem))
       
            self.canvas.stop_event_loop()
            self.plot_cmeans(self.FCM.data_points, self.FCM.centroids, self.FCM.memberships)
    
    def cluster_pause(self, interval):
        if self.canvas.figure.stale:
            self.canvas.draw()
        self.canvas.start_event_loop(interval)
        return
    
    def img_pause(self, interval):
        if self.canvas_img1.figure.stale:
                self.canvas_img1.draw()
        self.canvas_img1.start_event_loop(interval)
        return

############### points clustering ######################
    def stop_clustering(self):
        self.stop = True

    def run_clustering(self):
        self.canvas.stop_event_loop()
        self.stop = False
        for _ in range(self.FCM.epochs):
            self.FCM.one_step()
            self.epoch += 1
            self.plot_cmeans(self.FCM.data_points, self.FCM.centroids, self.FCM.memberships)
            if self.stop:
                break
    
    def reset_clustering(self):
        self.canvas.stop_event_loop()
        self.stop = True
        self.FCM.reset()
        self.epoch = 0
        self.plot_cmeans(self.FCM.data_points, self.FCM.centroids, self.FCM.memberships)


    def plot_cmeans(self, data, centers, membership, save_as=None):
        clusters_id = np.argmax(membership, axis=1)
        clusters_mem = np.max(membership, axis=1)
        data = data.T

        #fig, ax = plt.subplots()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.suptitle('Epoch: {}'.format(self.epoch), fontsize=16)
        scatter = ax.scatter(data[0], data[1], c=clusters_id, alpha=0.5,s=clusters_mem*100, label=clusters_id)
        #print(centers)
        
        ax.scatter(centers.T[0], centers.T[1], c=np.arange(centers.shape[0]), marker='X', edgecolor='black', lw = 1)
        labels = []
        for i in range(centers.shape[0]):
            labels.append(f"Cluster {i}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #plt.legend(handles=scatter.legend_elements()[0], labels=labels)
        if save_as is not None:
            plt.savefig(save_as+".png", tight_layout=True)
        self.canvas.draw()
        self.cluster_pause(0.2)

############### img segmentation ######################
    def run_segmentation(self):
        self.canvas_img1.stop_event_loop()
        self.stop_img = False
        self.active_img = 1
        for _ in range(self.FCM_img.epochs):
            self.FCM_img.one_step()
            self.epoch_img += 1
            self.plot_segmentation()
            if self.stop_img:
                break
    
    def stop_segmentation(self):
        self.stop_img = True

    def plot_segmentation(self):
        self.figure_img1.clear()
        ax = self.figure_img1.add_subplot(111)
        self.figure_img1.suptitle('Epoch: {}'.format(self.epoch_img), fontsize=16)
        if self.active_img == 0:
            ax.imshow(self.FCM_img.grey_img, cmap='gray')
        else:
            img = self.FCM_img.reconstruct_img()
            ax.imshow(img)
        
        self.canvas_img1.draw()
        self.img_pause(0.2)

    def switch(self):
        self.canvas_img1.stop_event_loop()
        self.active_img = 1 - self.active_img
        self.plot_segmentation()
    
    def reset_segmentation(self):
        self.canvas_img1.stop_event_loop()
        self.stop_img = True
        self.FCM_img.reset()
        self.active_img = 0
        self.epoch_img = 0
        self.plot_segmentation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/points/data_9c_100_mu_15_var_6.csv', help='path to data csv')
    parser.add_argument('--num_clusters', default=9, type=int, help='path to centroids data csv')
    parser.add_argument('--epochs', default=40, type=int, help='number of epochs for algorithm')
    parser.add_argument('--q', default=2, type=float, help='fuzziness')

    parser.add_argument('--data_path_img', default='../data/img/covid_01.jpeg', help='path to data csv')
    parser.add_argument('--num_clusters_img', default=11, type=int, help='path to centroids data csv')
    parser.add_argument('--epochs_img', default=40, type=int, help='number of epochs for algorithm')
    parser.add_argument('--q_img', default=2, type=float, help='fuzziness')

    parser.add_argument('--save_img', default="cmeans_plot", type=str, help='name of file to save cmeans plot')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    main = Window(args)
    main.resize(1200, 900)
    main.show()

    sys.exit(app.exec_())