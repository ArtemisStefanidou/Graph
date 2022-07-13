
# Import the necessary libraries
import cv2
import numpy as np

# Read the image as a grayscale image

pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148.tif'
#pathImage = pathFolder + nameImage
#gki eikona
img = cv2.imread( pathImage, cv2.IMREAD_GRAYSCALE)

# Threshold the image
ret,img = cv2.threshold(img, 127, 255, 0)

# Step 1: Create an empty skeleton
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    #Step 3: Substract open from the original image
    temp = cv2.subtract(img, open)
    #Step 4: Erode the original image and refine the skeleton
    #
    eroded = cv2.erode(img, element)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        break

# Displaying the final skeleton
#imS = cv2.resize(skel, (960, 540)) 
#cv2.imshow("Skeleton",imS)
cv2.imshow("Skeleton",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()