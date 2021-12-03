import sys
from PyQt5.QtWidgets import QDialog, QApplication, QGridLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget, QLabel

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
        

        self.FCM = FuzzyCMeans(args.task, args.data_path, args.epochs, args.num_clusters, args.q, args)
        self.stop = False
        # a figure instance to plot on
        self.figure, self.ax = plt.subplots()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        #self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Run')
        self.button.clicked.connect(self.run_clustering)
        self.button3 = QPushButton('Stop')
        self.button3.clicked.connect(self.stop_clustering)
        
        self.button2 = QPushButton('B2')
        self.button2.clicked.connect(self.a)
        # set the layout
        #layout = QVBoxLayout()
        #layout.addWidget(self.toolbar)
        #layout.addWidget(self.canvas)
        #layout.addWidget(self.button)
        #layout.addWidget(self.button2)
        #self.setLayout(layout)
        

        mainLayout = QGridLayout()
        vLayout1 = QVBoxLayout()
        self.tab1_1 = QWidget()
        self.tab1_1.layout = QVBoxLayout()
        #self.tab1_1.layout.addWidget(QLabel('dsalk;da'))
        self.tab1_1.layout.addWidget(self.canvas)
        self.tab1_1.layout.addWidget(self.button)
        self.tab1_1.layout.addWidget(self.button3)
        self.tab1_1.setLayout(self.tab1_1.layout)

        self.btn = QPushButton('A BUtton')
        self.btn.clicked.connect(lambda: print('Hello'))
        self.tab1_2 = QWidget()
        self.tab1_2.layout = QVBoxLayout()
        self.tab1_2.layout.addWidget(self.btn)
        self.tab1_2.setLayout(self.tab1_2.layout)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab1_1, 'tab1')
        self.tabs.addTab(self.tab1_2, 'tab2')
        mainLayout.addWidget(self.tabs, 0, 0)
        #mainLayout.addWidget(self.button)
        self.setLayout(mainLayout)
        cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)


        # init plots
        self.figure.canvas.draw_idle()
        self.figure.canvas.start_event_loop(0.001)
        self.plot_cmeans(self.FCM.data_points, self.FCM.centroids, self.FCM.memberships)


    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    
    def mypause(self, interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

    def a(self):
        print('ajfiopsa')

    def stop_clustering(self):
        self.stop = True

    def run_clustering(self):
        self.stop = False
        for i in range(self.FCM.epochs):
            self.FCM.one_step()

            self.plot_cmeans(self.FCM.data_points, self.FCM.centroids, self.FCM.memberships)
            if self.stop:
                break
            



    def plot_cmeans(self, data, centers, membership, save_as=None):
        clusters_id = np.argmax(membership, axis=1)
        clusters_mem = np.max(membership, axis=1)
        data = data.T

        #fig, ax = plt.subplots()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
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
        self.mypause(0.2)


    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = [random.random() for i in range(10)]

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # discards the old graph
        # ax.hold(False) # deprecated, see above

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/marek/Documents/FIT/mit/sfc/vut-fit-sfc/data/data_9c_100_mu_15_var_6.csv', help='path to data csv')
    parser.add_argument('--num_clusters', default=9, type=int, help='path to centroids data csv')
    parser.add_argument('--epochs', default=40, type=int, help='number of epochs for algorithm')
    parser.add_argument('--q', default=2, type=int, help='fuzziness')
    parser.add_argument('--task', default="points", type=str, help='cmeans on 2D data points or img')
    parser.add_argument('--save_img', default="cmeans_plot", type=str, help='name of file to save cmeans plot')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    main = Window(args)
    main.show()

    sys.exit(app.exec_())