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
import numpy as np
from skimage import io
from skimage.morphology import closing
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


def euclideanHeuristic(position1, position2):
    xy1 = position1
    xy2 = position2
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


# open and skeletonize
pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148.tif'
#pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148(1).tif'

#Αυτή η μέθοδος επιστρέφει μια εικόνα που έχει φορτωθεί από το καθορισμένο αρχείο
#cv2.IMREAD_GRAYSCALE: Καθορίζει τη φόρτωση μιας εικόνας σε λειτουργία κλίμακας του γκρι.
# Εναλλακτικά, μπορούμε να περάσουμε ακέραια τιμή 0 για αυτήν τη σημαία
img = cv2.imread( pathImage, cv2.IMREAD_GRAYSCALE)

#Ανώτερη τιμή κατωφλίου. Όλα τα εικονοστοιχεία με ένταση μεγαλύτερη από αυτήν την τιμή θεωρείται ότι είναι στο προσκήνιο
binary = img > filters.threshold_otsu(img)

#img = data.horse()
ske = skeletonize(binary).astype(np.uint16)
# print("the shape of the image",ske.shape)
# print("the size of the image",ske.size)


# build graph from skeleton
graph = sknw.build_sknw(ske, ('iso'==False))


# draw image
plt.imshow(img, cmap='gray')



# draw edges by pts
for (s,e) in graph.edges():
    #print((s,e))
    #to pts ta kanei hashable
    ps = graph[s][e]
    print("-------ps-------", ps)
    print("-------ps type----------", type(ps))
    plt.plot(ps[:,1], ps[:,0], 'green')



# draw node by o
nodes = graph.nodes()

# ps = np.array([nodes[i]['o'] for i in nodes])
ps = np.array([nodes[i]['o'] for i in nodes])

plt.plot(ps[:,1], ps[:,0], 'r.')
# print("Here is the graph edges",graph.edges())
#print("Here is the graph edges",graph.edges())
# print(ps)

#https://forum.image.sc/t/measuring-path-lengths-between-all-nodes-in-a-skeletonized-mesh-analyze-skeleton-2d-3d/11540/3


distances = []
for i in range(0, len(ps)-1):
    # print("nodes-->",ps[i])
    distances.append(((ps[i], ps[i+1]),euclideanHeuristic(ps[i], ps[i+1])))

# for i in range(0, len(ps)-1):
    # print("distances", distances[i])

# title and show
plt.title('Build Graph')
plt.show()