#!/usr/bin/python
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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


def plotter(stt_list,length):
    
    for i in length:
        ax.scatter(xs, ys, zs, c=c, marker=m)
        
        ax.set_xlabel('East')        
        ax.set_ylabel('North')
        ax.set_zlabel('Elevation')
        
    plt.show()

east = 9527000
north = 10574960
elevation = 100000
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 1000
c = 'g'
m = 'o'

global j
j = STT()

with open('stt_loc.csv', 'rb') as csvfile:
    tags = csv.reader(csvfile,delimiter=';', quotechar='"')
    for row in tags:
        j.add_tag(row[0])
        j.add_coord([int(row[1]) - east], [int(row[2]) - north], [int(row[3]) -  elevation])       

for line in j.coords:
    x = line.strip()[0]
       

#plotter(j,len(j.tag))
