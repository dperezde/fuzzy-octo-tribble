#!/usr/bin/python
from scipy import spatial
import csv
import numpy as np
from numpy import random, nanmax, argmax, unravel_index


east = 9527000
north = 10574960
elevation = 100000
with open('stt_loc.csv', 'rb') as csvfile:
    tags = csv.reader(csvfile, delimiter=';', quotechar = '"')

    coords = [[int(col[1]) - east, int(col[2]) - north, int(col[3]) - elevation]  for col in tags]
    arr = np.array([[c for c in row] for row in coords])

tree = spatial.KDTree(arr)
print tree.query(arr[0],40)

