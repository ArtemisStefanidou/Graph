# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:28:47 2022

@author: USER
"""

from skimage.morphology import skeletonize
from skimage import data
import sknw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters

# open and skeletonize
pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148.tif'
#pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148(1).tif'

img = cv2.imread( pathImage, 0)

binary = img > filters.threshold_otsu(img)

#img = data.horse()
ske = skeletonize(binary).astype(np.uint16)

# build graph from skeleton
graph = sknw.build_sknw(ske, ('iso'==False))


# draw image
plt.imshow(img, cmap='gray')

# draw edges by pts
#listEdges = [(0, 5), (1, 18), (2, 10), (3, 20), (4, 16), (6, 7), (8, 11), (9, 12), (13, 14), (14, 16), (14, 15), (16, 17), (17, 18), (17, 23), (18, 19), (19, 21), (19, 30), (20, 21), (21, 27), (22, 24), (23, 25), (23, 31), (24, 25), (24, 26), (25, 28), (29, 32)]
listEdges = [(0, 5), (1, 18), (2, 10) ,(3, 20), (4, 16), (6, 7), (8, 11), (9, 12), (13, 14), (14, 16), (14, 15), (17, 18)]
print("The len of listed is", len(listEdges))
for (s,e) in listEdges:
    ps = graph[s][e]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')

# draw node by o
nodes = graph.nodes()
print("nodes",nodes[0])
print("The len of nodes is", len(nodes))
#ps = np.array([nodes[i]['o'] for i in nodes])
#plt.plot(ps[:,1], ps[:,0], 'r.')
print("Here is the graph edges",listEdges)
#print("Here is the graph edges",graph.edges())

print(ps)

# title and show
plt.title('Build Graph')
plt.show()