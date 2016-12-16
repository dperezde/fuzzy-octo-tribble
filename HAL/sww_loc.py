#!/usr/bin/python
import csv
import numpy as np
from numpy import random, nanmax, argmax, unravel_index
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

class STT:
    def __init__(self):
        self.tag = []
        self.coords = [[], [], []]
        self.x = []
        self.y = []
        self.z = []

    def add_tag(self, tag):
        self.tag.append(tag)
    def add_coord(self, x, y, z):
        self.coords.append([x, y, z])
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)


def plotter(stt_list,length):
    
    for i in length:
        ax.scatter(xs, ys, zs, c=c, marker=n)
        
        ax.set_xlabel('East')        
        ax.set_ylabel('North')
        ax.set_zlabel('Elevation')
        
    plt.show()

def temp:
    east = 9527000
    north = 10574960
    elevation = 100000
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 1000
    c = 'g'
    n = 'o'

    global j
    j = STT()
    with open('stt_loc.csv', 'rb') as csvfile:
        tags = csv.reader(csvfile,delimiter=';', quotechar='"')
        

        coords = [[int(col[1]) - east, int(col[2]) - north, int(col[3]) - elevation] for col in tags]
        arr = np.array([[c for c in row] for row in coords])

#print arr

    D = pdist(arr)
    D = squareform(D)
#N, [I_row, I_col] = nanmax(D), unravel_index( argmax(D), D.shape)



    with open('stt_loc.csv', 'rb') as csvfile:
        tags = csv.reader(csvfile,delimiter=';', quotechar='"')
        tag = [col[0] for col in tags]

#print tag
    loop1 = list(D[0,:])

    for i in xrange(40):
        print loop1.index(min(loop1))
        loop1[loop1.index(min(loop1))]= max(loop1)
    




