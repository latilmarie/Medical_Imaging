#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:05:32 2023

@author: marie
"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import time

#%% Content of the code

##### Useful functions to call for processing the images
0. Print the image with OpenCV
1. Compute the histogram
2. Compute the segmentation with the OTSU method and threshold the image
3. Label the different components of an image with the connected components method

##### Processing the images for the project
4. Get the contours of the rice image
5. Compute the ratio of green pixels over red pixels on the RoBlood images
6. Separate the background from the anatomical information on the ultrasound image


#%% 0. Function to call in order to show an image with OpenCV

def iprint(image,title=" "):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#%% 1 Histogram function

def histogram (image):
    H = np.zeros((256))
    for i in range(256):
        H[i] = np.sum(image==i)
    # plt.figure()
    # plt.bar(np.arange(256), H)
    return H


#%% 2. Segmentation function - OTSU method

def segmentation(img):
    H = histogram(img)
    
    N = (np.shape(img)[0]*np.shape(img)[1])     # number of total pixels
    P = H/N                                     # probabilities pi for each pixel value
    muT = np.sum(np.arange(256)*P)              # total mean level
    sigma2=np.zeros((256))
    
    for k in range (256):
        mu = np.sum(np.arange(k+1)*P[:(k+1)])   # mean level
        omega = np.sum(P[:(k+1)])               # probability of the class occurrence
        sigma2[k] = (muT*omega - mu)**2 / (omega*(1-omega)) if omega != 0 and omega != 1 else 0 # considering the cases where the denominator can be equal to 0
    
    sigma_max = np.max(sigma2)                  # find the maximum
    k_opt = np.where(sigma2==sigma_max)[0][0]   # find the range of the maximum sigma
    print("Value of threshold:", k_opt)
    
    seg = np.copy(img)
    seg[seg<=k_opt] = 0                         # segmentation with the optimal k_opt
    seg[seg>k_opt] = 255
    iprint(seg) # uint8
    return seg

#%% 3. Connected components function

def connected_components(img, seg):
    if seg:                         # OTSU method, False if the threshold has been set by hand,
        img = segmentation(img)     # the image is uint8
        
    img = img.astype(int)           # the image is int64
    img[img==255]=-1                # regions of interest to -1
    img = np.pad(img, ((1,1)))      # zero-padding to avoid problems on the edges
    
    values = np.where(img==-1)      # all the coordinates where image == -1
    k=1                             # first label number
    classes = []                    # list of the label numbers
    
    for i in range (len(values[0])):
        a, b = values[0][i], values[1][i]   # coordinates of the -1 pixels in the image
        
        if img[a-1, b] <= 0 and img[a+1, b] <= 0 and img[a, b-1] <= 0 and img[a, b+1] <= 0: # any neighbours already labelled
            img[a,b]=k              # new label for the pixel
            classes.append(k)
            k=k+1                   # increase the class number for the next pixel
            
        else:
            img[a,b] = min( i for i in [img[a-1, b], img[a+1, b], img[a, b-1], img[a, b+1]] if i>0 ) # minimum of the labelled neighbours
            
            for da, db in ((-1,0), (1,0), (0,-1), (0,1)):                   # put all the neighbours already labelled to the same label
                if img[a+da, b+db] != img[a,b] and img[a+da, b+db] > 0:
                    classes.remove(img[a+da, b+db])
                    img[img==img[a+da,b+db]] = img[a,b]

    classes_sort = []               # sort the class labels
    for i in range (len(classes)):
        img[img==classes[i]] = i+1
        classes_sort.append(i+1)
        
    print("Number of components:", len(classes_sort))
    
    img = img[1:np.shape(img)[0]-1, 1:np.shape(img)[1]-1] # remove the zero padding added at the beginning
    img = img.astype(np.uint8) # uint8
    iprint(img)
        
    return img

#%% 4. Contours of the rice image - erosion and dilation

##### Read and print the rice image
path = "/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/Project 2"
os.chdir(path)

rice = cv2.imread("rice.tif")       # read the image
rice = rice[:,:,0]                  # get only 1 channel (the 3 are identical, it's a gray image)
iprint(rice)

##### Get the contours and print the image
seg = segmentation(rice)            # segmentation with OTSU method
# cv2.imwrite("/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/rice_seg.png", seg)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # little ellipse shape
contours = cv2.erode(seg, kernel)   # erosion
# cv2.imwrite("/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/rice_erode.png", contours)

contours = seg-contours             # get the contours
iprint(contours)
# cv2.imwrite("/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/rice_contours.png", contours)


#%% 5. Pixel ratio of the green over the red pixels in the RoBlood image

##### Read and print the RoBlood images
red = cv2.imread("RoBlood_man_thres.bmp", cv2.IMREAD_GRAYSCALE)                    # read the image
blue = cv2.imread("RoBlood_MathcedFilterResult.bmp", cv2.IMREAD_GRAYSCALE)         # read the image
iprint(red)
iprint(blue)

##### Change one value of the blue image in order to have 3 grey levels
blue[blue==255] = 100   # blue pixels become 100, background stay at 0

green = red-blue        # only red pixels stay at 255, 
                        # only blue pixels stay at 100, 
                        # commun red and blue pixels (green pixels) become 155 (255 (red)-100 (blue)), 
                        # commun background stay at 0
iprint(green)
# cv2.imwrite("/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/RoBloodGreen.png", green)

red_pixels = np.sum(red==255)       # count all the red pixels (including the green ones)
green_pixels = np.sum(green==155)   # count only the green pixels

ratio = green_pixels/red_pixels
print(np.round(ratio*100, 0), "%")


#%% 6. Separate the background from the anatomical information

##### Read and print the ultrasound image
ultras = cv2.imread("US00002S.tif", cv2.IMREAD_GRAYSCALE)  # read the image
iprint(ultras)

##### Crop the image around the region of interest to work on a smaller image
cut = ultras[180:400,200:530]
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/Cut.png', cut)



##### First pass to isolate the information on the left side
seg2 = np.copy(cut)
seg2[seg2<=30] = 0      # threshold by hand (OTSU method removes to much information)
seg2[seg2>30] = 255
# iprint(seg2)

connect = connected_components(seg2, False)
plt.figure(), plt.imshow(connect)   # to see the values of the background labels
connect[connect<=4] = 0             # remove the background
connect[connect>4] = 255            # put the anatomical information to 255
# iprint(connect)

back = np.copy(connect)
back[connect==255] = cut[connect==255]  # get back the original values of anatomical information
iprint(back)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/Info_left.png', back)



##### Second pass to isolate the information on the right side
right = np.copy(cut)
right[:, :180] = 0      # select the right side
# iprint(right)

seg3 = np.copy(right)
seg3[seg3>=45] = 255    # threshold by hand
seg3[seg3<45] = 0

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
ero = cv2.erode(seg3, kernel2)  # erosion to separate a little the information from the noise

ero = connected_components(ero, False) # segmentation: information and noise have different labels
plt.figure(), plt.imshow(ero)   # to see the values of the background labels
ero[ero==1] = 0                 # remove the background
ero[ero==38] = 0
ero[ero>1] = 255                # keep anatomical information
ero = cv2.dilate(ero, kernel2)  # get back the anatomical information lost by erosion
iprint(ero)

back[ero==255] = cut[ero==255]  # get back the original information
iprint(back)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/Info_right.png', back)



##### Final image not cropped
final = np.zeros((np.shape(ultras)))
final[180:400,200:530] = back
iprint(final)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 2/USFinal.png', final)


