#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:37:12 2023

@author: marie
"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import time


#%% 0. Function to call in order to show an image with OpenCV

def iprint(image,title=" "):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
#%% 1. Read and print the original image

path = "/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1"
os.chdir(path)

image = cv2.imread("Your_xray.png")                             # read the image
image = image[:,:,0]                                            # get only 1 channel (the 3 are identical, it's a gray image)
# image = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)     # other possibility to keep 1 channel
iprint(image, "Your xray")                                      # print the image


#%% 2. Histogram computation

def histogram(img):   
    N = np.shape(img)[0]
    M = np.shape(img)[1]
    H = np.zeros((256,1))
    for i in range (N):
        for j in range (M):
            if img[i,j] < 0:  # to ignore the negativ values, useful for the stretched image
                continue
            else:
                x = img[i,j]
                H[x,0] = H[x,0]+1
    plt.bar(np.arange(np.size(H)), H[:,0])
    return H

H = histogram(image)

plt.bar(np.arange(np.size(H)), H[:,0])
# save the figure
# plt.savefig('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Histogram.png', dpi=300, bbox_inches='tight')


#%% 3. Image stretch

image_stretch = np.copy(image)          # to not change the original image
image = image.astype(int)
image[image==8]=0
image[image>240]=0                      # putting the unwanted values together at 0

image_stretch = (255*
                 (image - np.min(image[np.nonzero(image)])) / 
                 (image.max()-np.min(image[np.nonzero(image)]))
                 ).astype(int)          # streching the image, without the unwanted data in the dynamic range (ignored thanks to np.nonzero, which takes the second minimum)

# image_stretch[image_stretch<0]=0      # the unwanted data still changed, became negativ, don't forget to put them to 0 if we want to visualize the image

H_stretch = histogram(image_stretch)    # the unwanted values are not taken into acount for the histogram computation
plt.figure()
plt.bar(np.arange(np.size(H_stretch)), H_stretch[:,0])
# plt.savefig('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Histogram_stretch.png', dpi=300, bbox_inches='tight')


#%% 4. Invert image

image_inv = -(image-255)

iprint(image_inv)

# to save the image
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_invert.png', image_inv)

#%% 5. Filtering in the time domain 

def filtering(img, kernel, output):
    N = np.shape(img)[0]
    M = np.shape(img)[1]
    K = np.shape(kernel)[0]
    for i in range (0, N-K):
        for j in range (0, M-K):
            output[i+K//2, j+K//2] = np.sum(img[i:i+K, j:j+K]*kernel)
    return (output)

### Filtering 3x3
m=3
filt = np.ones((m,m))/m**2  # classic 3x3 filter
image_conv3 = np.copy(image)        # to not change the original image

tic = time.time()
image_conv3 = filtering(image, filt, image_conv3)
tac = time.time()
print(np.round(tac-tic,2), "s")    # print the time to compute the filtered image

iprint(image_conv3)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_filt'+str(m)+'_conv.png', image_conv3) # to save the image


### Filtering 5x5
m=5
filt = np.ones((m,m))/m**2  # classic 5x5 filter
image_conv5 = np.copy(image)        # to not change the original image

tic = time.time()
out = filtering(image, filt, image_conv5)
tac = time.time()
print(np.round(tac-tic,2), "s")

iprint(image_conv5)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_filt'+str(m)+'_conv.png', image_conv5) # to save the image


#%% 6. Filtering in the frequency domain

def frequency(img, m, output): # m is the kernel size wanted
    (N,M) = np.shape(image)
    kernel = np.zeros((N, M))
    kernel[N//2 - (m-1)//2 : N//2 + ((m-1)//2)+1 , M//2 - (m-1)//2 : M//2 + ((m-1)//2)+1] = 1/m**2

    output = np.abs(
                np.fft.fftshift(
                np.fft.ifft2(
                np.fft.fft2(img)*np.fft.fft2(kernel))
                )).astype(np.uint8)
    return(output)

### Filtering 3x3
image_freq3 = np.copy(image)
m=3

tic = time.time()
image_freq3 = frequency(image, m, image_freq3)
tac = time.time()
print(np.round(tac-tic,2), "s")

iprint(image_freq3)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_freq'+str(m)+'.png', image_freq3)


### Filtering 5x5
image_freq5 = np.copy(image)

tic = time.time()
image_freq5 = frequency(image, 5, image_freq5)
tac = time.time()
print(np.round(tac-tic,2), "s")

iprint(image_freq5)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_filt'+str(m)+'.png', image_freq5)


#%% 7. Gaussian OpenCV function

image_gauss = np.copy(image)
image_gauss = cv2.GaussianBlur(image, (5,5), 0)
iprint(image_gauss)

# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_gauss5.png', image_gauss)


#%% 8. Sharpen the image

filt = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

### Sharpen the image blurred by the 3x3
image_sharp3 = np.copy(image_conv3)

image_sharp3 = filtering(image_conv3, filt, image_sharp3)
iprint(image_sharp3)

# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_sharp.png', image_sharp3)


### Sharpen the image blurred by the 5x5
image_sharp5 = np.copy(img_conv5)

image_sharp5 = filtering(img_conv5, filt, image_sharp5)
iprint(image_sharp5)

# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/Image_sharp5.png', image_sharp5)

#################
#################

#%% 9. Improve the heart image

heart = cv2.imread('Heart.png')
heart = heart[:,:,0]                # to keep 1 channel only
heart = heart[:2300, 710:3710]      # to cut the image in order to have a smaller one -> faster to compute

iprint(heart)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/heart_cut.png', heart)


#%% 9.1 Gaussian Blur followed by sharpen filter -> not improving the image

heart_gauss = cv2.GaussianBlur(heart, (25,25), 0)
iprint(heart_gauss)

filt = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
heart_sharp = np.copy(heart_gauss)
heart_sharp = filtering(heart_gauss, filt, heart_sharp)
iprint(heart_sharp)


#%% 9.2 Threshold

heart_th = np.copy(heart)

### threshold 1
heart_th[heart_th>140] = 255
heart_th[heart_th<=120] = 0

iprint(heart)
iprint(heart_th)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/heart_tresh_120_140.png', heart_th)

### threshold 2
heart_th2[heart_th2>140] = 255
heart_th2[heart_th2<=90] = 0

iprint(heart)
iprint(heart_th2)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/heart_tresh_90_140.png', heart_th2)


#%% 9.3 Threshold by intensity-level slicing

heart_sl = np.copy(heart)
heart_sl = heart_sl.astype(int)

heart_sl[heart_sl>170] = heart_sl[heart_sl>170]+30
heart_sl[heart_sl>250] = heart_sl[heart_sl>250]-30      # add 30 to the pixels between 170 and 220

heart_sl[heart_sl<120] = heart_sl[heart_sl<120]-50
heart_sl[heart_sl<0] = heart_sl[heart_sl<0]+50          # remove 50 to the pixels between 50 and 120

heart_sl = heart_sl.astype(np.uint8)                    # convert the image to plot it

heart_sl[heart_sl<=45] = 0
iprint(heart)
iprint(heart_sl)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/heart_slice.png', heart_sl)

#%% 9.3.2 Apply a Gaussian blur to the sliced image (and sharpen afterwards) -> not improving

# heart_gauss_sl = cv2.GaussianBlur(heart_sl, (25,25), 0)
# iprint(heart_gauss_sl)

# heart_gauss_sl[heart_gauss_sl<=50] = 0
# iprint(heart_gauss_sl)

# filt = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# heart_sharp_sl = np.copy(heart_gauss_sl)

# heart_sharp_sl = filtering(heart_gauss_sl, filt, heart_sharp_sl)
# iprint(heart_sharp_sl)


#%% 9.4 Gamma power transformation

### gamma = 1.5
gamma = 1.5
heart_gamma1 = np.array(255*(heart / 255) ** gamma, dtype = 'uint8') 

iprint(heart)
iprint(heart_gamma1)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/heart_gamma_1_5.png', heart_gamm1)


### gamma = 2
gamma = 2
heart_gamma2 = np.array(255*(heart / 255) ** gamma, dtype = 'uint8') 

iprint(heart)
iprint(heart_gamma2)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/heart_gamma_2.png', heart_gamm2)


### threshold of 30 on the filtered image by gamma=2 followed by a gamma power with gamma = 0.7
heart_gamma2[heart_gamma2<30] = 0
gamma=0.7

heart_gamma3 = np.array(255*(heart_gamma2 / 255) ** gamma, dtype = 'uint8') 
iprint(heart_gamma3)
# cv2.imwrite('/Users/marie/Documents/ENSE3/3A/Courses/Medical Imaging/Mini Project 1/heart_gamma_2_07.png', heart_gamm3)


#%% Function to compare 2 images

def difference(img1, img2):
    img1 = img1.astype(int)
    img2 = img2.astype(int)
    diff = np.abs(img1-img2).astype(np.uint8)
    iprint(diff)
    return (diff)

diff_method = difference(image_conv3, image_freq3)

