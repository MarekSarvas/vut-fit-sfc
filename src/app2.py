import sys
from PyQt5.QtWidgets import QDialog, QApplication, QGridLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget, QLabel

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random

class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        #self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)
        
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

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

        


    def a(self):
        print('ajfiopsa')
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
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())