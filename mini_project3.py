#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:21:15 2023

@author: marie
"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
# import time

#%% Content of the code

##### Useful functions to call for processing the images
0. Print the image with OpenCV
1. Compute the histogram
2. Compute the segmentation with the OTSU method and threshold the image
3. Label the different components of an image with the connected components method
4. Adaptive thresholding function

##### Processing the images for the project
5. Isolate the ribs 
6. Count the number of ribs
7. Create a solid area
    Compute is features in a vector:
    7a. Compute the area
    7b. Compute the circumference
    7c. Compute the height and width
8. Create an image with the red solid part and the green pattern

#%% 0. Function to call in order to show an image with OpenCV

def iprint(image,title=" "):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

#%% 1 Histogram function

def histogram (image, plot):
    H = np.zeros((256))
    for i in range(256):
        H[i] = np.sum(image==i)
    if plot:
        plt.figure()
        plt.bar(np.arange(256), H)
    return H


#%% 2. Segmentation function - OTSU method

def segmentation(img):
    H = histogram(img, False)
    
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
        simg = segmentation(img)     # the image is uint8
        
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

#%% 4. Adaptive thresholding function

def adapt_thresh(img, ker, cst, out):
    img = img.astype(float)
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            # Calculate the region of interest (ROI)
            i1, i2, j1, j2 = max(0, i-ker// 2), min(M, i+ker// 2), max(0, j-ker// 2), min(N, j+ker// 2)
            roi = img[i1:i2, j1:j2]
            
            # Calculate the local threshold
            threshold = np.mean(roi)-cst
            
            # Apply the threshold to the pixels
            if img[i, j] > threshold:
                out[i, j] = 255
            else:
                out[i, j] = 0
    out = out.astype(np.uint8)


#%% 5. Isolate the ribs

##### Read and print the X-ray image
path = "/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/Project 3"
os.chdir(path)

xray = cv2.imread("XRAY.tif", cv2.IMREAD_GRAYSCALE)   # read the image
iprint(xray)


##### Adaptive thresholding
adapt = np.copy(xray)
adapt_thresh(adapt, 30, 8, adapt)

iprint(adapt)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/ribs_1.png', adapt)


##### Segmentation with OTSU method
seg = segmentation(xray)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/ribs_seg.png', seg)


##### Apply the segmentation as a mask to get only the lung region
adapt[seg==255]=0
iprint(adapt)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/ribs.png', adapt)


##### Apply morphological operations to get rid of the noise between the ribs
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19,9))
ero = cv2.erode(adapt, kernel)
ero = cv2.dilate(ero, kernel)
iprint(ero) 
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/ribs_erode_2.png', ero)


                
#%% 6. Count the number of ribs

num = np.copy(ero)
num = connected_components(num, False)
plt.imshow(num)

##### Remove the components with less than 300 pixels in them
k=1
for i in range(1, np.max(num)+1):
    if np.sum(num==i) < 300:
        num[num==i] = 0
    else: 
        num[num==i] = k
        k=k+1

plt.figure(), plt.imshow(num)
print("Number of components:", np.max(num))

# plt.savefig('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/ribs_count_3.png', dpi=300, bbox_inches='tight')


#%% 7. Create a solid area

##### Read and print the redcell image
path = "/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/Project 3"
os.chdir(path)

red = cv2.imread("cell_red.tif", cv2.IMREAD_GRAYSCALE)       # read the image
iprint(red)

##### Segmentation with the OTSU method
seg = np.copy(red)
seg = segmentation(seg)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/red_seg.png', seg)


##### Morphological operations to remove the noise outside of the red cell part
struct = np.copy(seg)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
struct = cv2.dilate(struct, kernel)         # dilation to connect the pixels of interest

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
struct = cv2.erode(struct, kernel)          # erosion to remove the noise outside of the region of interest

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
struct = cv2.dilate(struct, kernel)         # dilation to get back the information from the region of interest

iprint(struct)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/red_seg_ero.png', struct)


##### Connected components to keep the cell part and remove the last noise
struct = connected_components(struct, False)
plt.imshow(struct)

##### Look for the largest component to keep it
large = np.zeros((2,np.max(struct)+1))
for i in range(1, np.max(struct)+1):
    large[1,i]= np.sum(struct==i)
    large[0,i] = i

anat=np.argmax(large[1,:]) # get the class label of the cell

red2 = np.copy(red)
red2[struct!=anat] = 0
iprint(red2)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/red_noise.png', red2)


#%% Method 1 - Segmentation with OTSU method

##### OTSU method
seg2 = np.copy(red2)
seg2 = segmentation(seg2)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/red_seg3.png', seg2)


##### Morphological to create the solid area
struct2 = np.copy(seg2)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
struct2 = cv2.dilate(struct2, kernel2)
iprint(struct2)

struct2 = cv2.erode(struct2, kernel2)
iprint(struct2)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/red_solid.png', struct2)


##### Compare method 1 result to the original image without noise
test = cv2.imread("cell_green.tif")
test[:,:,0] = 0
test[:,:,2] = struct2
test[:,:,1] = red2
iprint(test)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/comp1.png', test)


#%% Method 2 - Segmentation with a threshold at 0

##### Treshold at 0
seg3 = np.copy(red2)
seg3[seg3!=0]=255
iprint(seg3)

##### Morphological operation to create the solid area
struct3 = np.copy(seg3)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
struct3 = cv2.dilate(struct3, kernel2)
iprint(struct3)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
struct3 = cv2.erode(struct3, kernel2)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/red_solid_2_2.png', struct3)


##### Compare method 2 to the original image without noise
test = cv2.imread("cell_green.tif")
test[:,:,0] = 0
test[:,:,2] = struct3
test[:,:,1] = red2
iprint(test)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/comp2.png', test)


#%% 7a. Create a feature vectore and compute the area

##### Create a feature to stock the 4 variables: area, circumference, height and width
feature = np.zeros((4,1))

##### Fill the area
feature[0,0]=np.sum(struct2==255)


#%% 7b. Compute the circumference

##### Get the contours of the solid area
area = np.copy(struct2)
ero = np.copy(struct2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
ero = cv2.erode(ero, kernel)

area = area-ero

iprint(area)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/circum.png', area)


##### Count only the pixels of the extern contour using connected-components method
area2 = np.copy(area)
area2 = connected_components(area2, False)
plt.imshow(area2)

large = np.zeros((2,np.max(area2)+1))
for i in range(1, np.max(area2)+1):
    large[1,i]= np.sum(area2==i)
    large[0,i] = i
    
circum = np.max(large[1,:])


##### Fill the feature vector
feature[1,0] = circum

#%% 7c. Compute the height and width

##### Get the coordinates of pixel values of 255
values = np.where(area==255)
P = len(values[0])


##### Get the coordinates of the four points of interest: up, down, left and right
i1, j1 = values[0][0], values[1][0]
i2, j2 = values[0][P-1], values[1][P-1]

j3 = np.min(values[1])
i3 = np.where(values[1]==j3)[0][0]

j4 = np.max(values[1])
i4 = np.where(values[1]==j4)[0][0]


##### Compute the height
a = i2-i1
b = j2-j1
c = np.round(np.sqrt(a**2+b**2),0)


##### Compute the width
d = i4-i3
e = j4-j3
f = np.round(np.sqrt(d**2+e**2),0)


##### Fill the feature vector
feature[2:,0] = c,f


#%% 8. Create a color image with the red solid area and the green pattern

final = np.copy(struct2)

path = "/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/Project 3"
os.chdir(path)

green = cv2.imread("cell_green.tif")       # read the image
iprint(green)

green[:,:,0] = 0            # put the blue channel to 0
green[:,:,2] = struct2      # put the red solid area into the red channel

iprint(green)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 3/red_green.png', green)
