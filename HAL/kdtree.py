#!/usr/bin/python
from scipy import spatial
import csv
import numpy as np
from numpy import random, nanmax, argmax, unravel_index
from collections import Counter

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def gen_arr():
    east = 9527000
    north = 10574960
    elevation = 100000
    with open('stt_loc.csv', 'rb') as csvfile:
        tags = csv.reader(csvfile, delimiter=';', quotechar = '"')

        coords = [[int(col[1]) - east, int(col[2]) - north, int(col[3]) - elevation]  for col in tags]

        arr = np.array([[c for c in row] for row in coords])

    return arr

def give_me_loops():

    arr = gen_arr()


    loops = [[]*40]

    for i in xrange(764):
        l = [val for sublist in loops for val in sublist]
#    print l
        if i in l:
            continue
        else:
#        print len(arr)
            tree = spatial.KDTree(arr)
            d, loop = tree.query(arr[i],40, distance_upper_bound = float("inf"))
#            print loop
#            raw_input("Pause")
       # print len(loop)
            loops.append(loop)
            for el in loop:
                arr[el,:] = float("inf")


        a = [k for k,v in Counter(l).items() if v > 1]
#        print a

    loops.pop(0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    n = 1000
    c = 'g'
    n = 'o'

    array = gen_arr()
    for j in xrange(20):
        for i in loops[j]:
            ax.scatter(array[i][0], array[i][1], array[i][2], c = np.random.rand(3,1))

    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Elevation')
#    for i in loops[0]:
#        ax.scatter(array[i][0], array[i][1], array[i][2], c=c, marker = n)
#
#        ax.set_xlabel('East')
#        ax.set_ylabel('North')
#        ax.set_zlabel('Elevation')
#
#   for i in loops[1]:
#        ax.scatter(array[i][0], array[i][1], array[i][2], c= 'r', marker = '^')
#
#        ax.set_xlabel('East')
#        ax.set_ylabel('North')
#        ax.set_zlabel('Elevation')
#
#    for i in loops[2]:
#        ax.scatter(array[i][0], array[i][1], array[i][2], c= 'b', marker = 'o')
#
#        ax.set_xlabel('East')
#        ax.set_ylabel('North')
#        ax.set_zlabel('Elevation')
#
#    for i in loops[3]:
#        ax.scatter(array[i][0], array[i][1], array[i][2], c= 'y', marker = 'o')
#
#        ax.set_xlabel('East')
#        ax.set_ylabel('North')
#        ax.set_zlabel('Elevation')
#
#    for i in loops[4]:
#        ax.scatter(array[i][0], array[i][1], array[i][2], c= 'c', marker = 'o')
#
#        ax.set_xlabel('East')
#        ax.set_ylabel('North')
#        ax.set_zlabel('Elevation')

    plt.show()


give_me_loops()
