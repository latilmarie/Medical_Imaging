#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:29:05 2023

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

##### Processing the images for the project
5. Segmentation of the CT and PET images 
6. Get the coordinates of the landmarks
7. Apply the transformation to the PET image
8. Apply the new PET image as a mask to the CT image
9. Automatic registration on retina images

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


#%% 4. Segmentation of the CT and PET images

########## CT image ##########

##### Read and print the CT image
path = "/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/Images"
os.chdir(path)

ct = cv2.imread("CT.png", cv2.IMREAD_GRAYSCALE)   # read the image
iprint(ct)

##### Set a threshold to segmentate the image
ct_tresh = np.copy(ct)
thresh = 100
ct_tresh[ct_tresh>thresh]=255
ct_tresh[ct_tresh<=thresh]=0
iprint(ct_tresh)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/ct_tresh.png', ct_tresh)

##### Connected-component method to select the most classes
ct_con = connected_components(ct_tresh, False)
fig=plt.figure(), plt.imshow(ct_con)

##### Look for the largest components to keep it
large = np.zeros((2,np.max(ct_con)+1))
for i in range(1, np.max(ct_con)+1):
    large[1,i]= np.sum(ct_con==i)
    large[0,i] = i

vert = np.argmax(large[1,:]) # get the class label of the cell

##### Select the most components to keep on the image
ct_seg = np.copy(ct_con)
ct_seg[ct_con==vert] = 255
ct_seg[ct_con==17] = 255
ct_seg[ct_seg!=255] = 0
iprint(ct_seg)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/ct_seg.png', ct_seg)



########## PET image ##########

pet = cv2.imread("PET.png", cv2.IMREAD_GRAYSCALE)   # read the PET image
iprint(pet)

##### Set a threshold to the image
pet_seg = np.copy(pet)
pet_seg[pet_seg>70] = 255
pet_seg[pet_seg<=70] = 0
iprint(pet_seg)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/pet_seg.png', pet_seg)


##### Try with OTSU method
pet_seg2 = np.copy(pet)
pet_seg2 = segmentation(pet_seg2)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/pet_seg2.png', pet_seg2)


#%% 6. Get the coordinates of the landmarks

points_ct = []
points_pet = []

img1 = np.copy(ct_seg)
img2 = np.copy(pet_seg)

##### Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_ct.append((x, y))
        cv2.circle(img1, (x, y), 30, 150, -1) # to draw the point on the image
        cv2.imshow("Image 1", img1)
        # cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/ct_key.png', img1)

def mouse_callback_img2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_pet.append((x, y))
        cv2.circle(img2, (x, y), 30, 150, -1)
        cv2.imshow("Image 2", img2)
        # cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/pet_key.png', img2)

        
##### Image 1 (CT)
cv2.namedWindow("Image 1") # Create windows for the image
cv2.setMouseCallback("Image 1", mouse_callback) # Set the callback function
cv2.imshow("Image 1", img1) # Display image
cv2.waitKey(0)
cv2.destroyAllWindows()

##### Image 2 (PET)
cv2.namedWindow("Image 2")
cv2.setMouseCallback("Image 2", mouse_callback_img2)
cv2.imshow("Image 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

##### Display the coordinates points
print("Points in Image 1 (CT):", points_ct)
print("Points in Image 2 (PET):", points_pet)

#%% 7. Apply transformation to the PET image

# The system is Ah = B

##### Define the matrix A
A = np.zeros((len(points_ct)*2, 8))

for i in range (len(points_ct)):
    A[2*i,:] = [points_pet[i][0], points_pet[i][1], 1, 0, 0, 0, -points_pet[i][0]*points_ct[i][0], -points_pet[i][1]*points_ct[i][0]]
    A[2*i+1,:] = [0, 0, 0, points_pet[i][0], points_pet[i][1], 1, -points_pet[i][0]*points_ct[i][1], -points_pet[i][1]*points_ct[i][1]]

##### Define the matrix B
B = np.zeros((len(points_ct)*2,1))

for i in range (len(B)//2):
    B[2*i,0] = points_ct[i][0]
    B[2*i+1,0] = points_ct[i][1]
    

##### Solve the systm
### Using pseudo-inverse with pinv
pinv_A = np.linalg.pinv(A) # Pseudo-inverse of A    
h = np.dot(pinv_A, B) # Solution

### Using SVD
# svd = np.linalg.svd(A) # Singular Value Decomposition
# pinvA = np.dot(np.dot(svd[2].T, np.diag(1. / svd[1])), svd[0].T)
# h = np.dot(pinv_A, B)
    
### Using solve
# h = np.linalg.solve(A, B)


##### Get the transformation matrix H
H = np.reshape(np.append(h, 1), (3, 3))
H = H.astype(np.float32)


##### Apply the perspective transformation
output_size = (ct_seg.shape[1], ct_seg.shape[0]) # Define the size of the output image
pet_transf = cv2.warpPerspective(pet_seg, H, output_size)
iprint(pet_transf)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/pet_transf.png', pet_transf)


#%% 8. Apply the new PET image as a mask to the CT image

final = np.copy(ct)
final[pet_transf==0] = 0
iprint(final)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/ct_extract.png', final)



#%% 9. Automatic registration on the retina images

##### Read and print the retina images
path = "/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/Images"
os.chdir(path)

retina1 = cv2.imread("retina1.png", cv2.IMREAD_GRAYSCALE)   # read the image
iprint(retina1)

retina2 = cv2.imread("retina5.png", cv2.IMREAD_GRAYSCALE)   # read the image
iprint(retina2)

##### Automatic registration - SIFT method
img1 = np.copy(retina1)
img2 = np.copy(retina2)


##### Set the minimum Hessian value
minHessian = 800

##### Initialize ORB detector
orb = cv2.ORB_create(minHessian)

##### Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

##### Draw keypoints on the images
img_keypoints_1 = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
img_keypoints_2 = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)

##### Show the images with keypoints
iprint(img_keypoints_1)
iprint(img_keypoints_2)


##### Step 2: Calculate descriptors (feature vectors)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.compute(img1, keypoints1)
keypoints2, descriptors2 = orb.compute(img2, keypoints2)

##### Convert descriptors to CV_32F
descriptors1 = descriptors1.astype(np.float32)
descriptors2 = descriptors2.astype(np.float32)


##### Step 3: Matching descriptor vectors using FLANN matcher
matcher = cv2.FlannBasedMatcher()
matches = matcher.match(descriptors1, descriptors2)

##### Drawing the results
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)

##### Show the matches
iprint(img_matches)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/retina_matches.png', img_matches)


##### Quick calculation of max and min distances between keypoints
max_dist = 0
min_dist = 100

for match in matches:
    dist = match.distance
    if dist < min_dist:
        min_dist = dist
    if dist > max_dist:
        max_dist = dist

print("Max dist :", max_dist)
print("Min dist :", min_dist)


##### Use only "good" matches (i.e. whose distance is less than 3*min_dist)
good_matches = []
for match in matches:
    if match.distance < 2.5 * min_dist:
        good_matches.append(match)

print("Good matches:", len(good_matches))

# Draw only the good matches
img_good_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
iprint(img_good_matches)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 4 - Final project/retina_good_matches.png', img_good_matches)

