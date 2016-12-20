#!/usr/bin/python3
from scipy import spatial
import csv
import numpy as np
from numpy import nanmax, argmax, unravel_index
from collections import Counter

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from functools import reduce
import math
import random

import operator

iteration_points = []

iteration_centroids = []
def gen_arr():
    east = 9527000
    north = 10574960
    elevation = 100000
    with open('stt_loc.csv', 'rt') as csvfile:
        tags = csv.reader(csvfile, delimiter=';', quotechar = '"')

        coords = [[float(col[1]) - east, float(col[2]) - north, float(col[3]) - elevation]  for col in tags]

        arr = [[c for c in row] for row in coords]

    return arr

def getDistance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    Note: This can be very slow and does not scale well
    '''
    print ("a.n is ", a.n, " and b.n is ", b.n)
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")
    
    ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)


class STT:
    def __init__(self):
        self.tag = []
        self.coords = [[], [], []]
        self.x = []
        self.y = []

class Point:
    '''
    An point in n dimensional space
    '''
    def __init__(self, coords):
        '''
        coords - A list of values, one per dimension
        '''
        
        self.coords = coords
        self.n = len(coords)
        
    def __repr__(self):
        return str(self.coords)


class Cluster:
    '''
    A set of points and their centroid
    '''
    
    def __init__(self, points, cluster_size):
        '''
        points - A list of point objects
        '''
        
        if len(points) == 0: raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points
        
        # The dimensionality of the points in this cluster
        self.n = points[0].n
        
        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n: raise Exception("ILLEGAL: wrong dimensions")
            
        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

        self.p_dist = {}

        self.clust_size = cluster_size

    def append_dist(self, d):
        self.p_dist.append(d)

    def pop_dist(self, d):
        self.p_dist_arr.pop(d)
   
    def get_clust_size(self):
        return self.clust_size

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)
    
    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid) 
        return shift
    
    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        
        return Point(centroid_coords)


def kmeans2(points, k, cutoff, cluster_size):
    #Choose random initial centroids
    initial = random.sample(points, k)

    # Create k clusters, one for each centroid
    clusters = [Cluster([p], cluster_size) for p in initial]

    for c in clusters:
        pos = 0
        for p in points:
            p_dist = {}
            p_dist[pos] = getDistance(p, clusters[0].centroid)

            p_dist_sort = sorted(p_dist.items(), key = operator.itemgetter(1))

            c.p_dist = p_dist_sort[:40]

            print (c.p_dist)





def kmeans(points, k, cutoff, cluster_size):
    global max_loop 
    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)
#    print(initial) 
    # Create k clusters using those centroids
    clusters = [Cluster([p], cluster_size) for p in initial]
    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1

        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getDistance(p, clusters[0].centroid)
            clusters[0].append_dist(smallest_distance)
        
            # Set the cluster this point belongs to
            clusterIndex = 0
        
            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getDistance(p, clusters[i+1].centroid)
                if len(clusters[i+1].p_dist) < clusters[i+1].clust_size:
                    max_loop = 0
                    # If it's closer to that cluster's centroid update what we
                    # think the smallest distance is, and set the point to belong
                    # to that cluster
                    if distance < smallest_distance:
                        smallest_distance = distance
                        clusterIndex = i+1

                        clusters[i+1].p_dist.append(distance)
                else:
                    max_loop = 1
                    if distance < smallest_distance:
                        for d in clusters[i+1].p_dist:
                            if max(distance, d) != distance:
                                print(clusters[i+1].p_dist)
                                print(d)
                                print(clusters[i+1].p_dist.index(d))
                                clusters[i+1].p_dist.remove(d)
                                print(clusters[i+1].p_dist)
                                smallest_distance = distance
                                clusterIndex = i+1

            if max_loop == 0:
                lists[clusterIndex].append(p)
        
        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # store the iteration's cluster points and respective centroids
        # the first iteration will only have centroids in the cluster
        if loopCounter >= 2:
            global iteration_points
            global iteration_centroids

            iter_point_data = []
            iter_centroid_data = []

            for j in range(k):
                x, y, z = [], [], []

                for p in clusters[j].points:
                    x.append(p.coords[0])
                    y.append(p.coords[1])
                    z.append(p.coords[2])

                c = clusters[j].calculateCentroid()

                iter_point_data.append([x, y, z])
                iter_centroid_data.append([c.coords[0], c.coords[1], c.coords[2]])

            iteration_points.append(iter_point_data)
            iteration_centroids.append(iter_centroid_data)

        # As many times as there are clusters ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print ("Converged after {:d} iterations".format(loopCounter-1))
            break

    return clusters, loopCounter-1

def round_up_arr(arr, size):
    i = size - len(arr)
    if i > 0:
        while i > 0:
            arr.append(arr[len(arr)-1])
            i -= 1

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

    plt.show()





def main():
    dimensions = 3

    n_clusters = 20

    opt_cutoff = 0.01

    arr = gen_arr()

    cluster_size = 40

#    arr = round_up_arr(arr, cluster_size * n_clusters)

    points = [Point(arr[i]) for i in range(len(arr))]

    kmeans2(points, n_clusters, opt_cutoff, cluster_size) 
    exit

    clusters, iterations = kmeans(points, n_clusters, opt_cutoff, cluster_size)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    colors = ['r','w', 'k','m', 'c', 'g','y', 'b']
    c = 'g'
    n = 'o'

    for i,cl in enumerate(clusters):
        color_sel = colors[np.random.randint(0,7)]
        for p in cl.points:
            print (" Cluster: ", i, "\t Point:", p, "\t Cluster size:", len(cl.points))
            col = colors[(i%7)]
            if i == 0:
                col = colors[0]
            if i == 1:
                col = colors[1]
            ax.scatter(p.coords[0], p.coords[1], p.coords[2], c = col)
            #if i > 8:
            #    break



    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Elevation')
    
    plt.show()



if __name__ == "__main__":
    main()
