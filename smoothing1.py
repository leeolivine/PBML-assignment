'''
Python 3.9.16v

date: 2023-04-11, Tues
author: You Ho Yeong, Lee Ga Ram

### Spatial Filtering ###
'''
#%% Import Library

import numpy as np
import cv2
import matplotlib.pyplot as plt

#%% Function

def spnoise(image, prob=0.05):
    output = np.copy(image)
    salt = np.ceil(prob * image.size * 0.5)
    pepper = np.ceil(prob * image.size * 0.5)

    # Add salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, int(salt)) for i in image.shape[:2]]
    output[tuple(coords) + (slice(None),)] = 255

    # Add pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, int(pepper)) for i in image.shape[:2]]
    output[tuple(coords) + (slice(None),)] = 0

    return output

#%% Load Image

img = cv2.imread(r'/Users/hoyeong/Desktop/python/PBML/Image/Kids_park.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#%% add salt & pepper effect

img_noise = spnoise(img, prob=0.05)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('on')

# Display the image with salt & pepper noise in the second subplot
axes[1].imshow(img_noise)
axes[1].set_title('Salt & Pepper Noise')
axes[1].axis('on')

#plt.tight_layout()
plt.show()

#%% Histogram

plt.figure(figsize=(10, 5))

# Total histogram (all channels combined)
plt.hist(img.ravel(), bins=256, color='orange', label='Total')

# Red channel histogram
# alpha stands for transparency
plt.hist(img[:, :, 0].ravel(), bins=256,
         color='red', alpha=0.5, label='Red')

# Green channel histogram
plt.hist(img[:, :, 1].ravel(), bins=256,
         color='green', alpha=0.5, label='Green')

# Blue channel histogram
plt.hist(img[:, :, 2].ravel(), bins=256,
         color='blue', alpha=0.5, label='Blue')

plt.xlabel('Value')
plt.ylabel('Hist')
plt.legend()
plt.show()

#%% Smoothing filter
'''
Improve Moving average and Gaussian filter 
by increasing kernel_size or sigma values
'''
kernel_size = 9
img_avg = cv2.blur(img_noise, (kernel_size, kernel_size))

# Gaussian Filter
sigma = 1
img_gaussian = cv2.GaussianBlur(img_noise, (kernel_size, kernel_size), sigma)

# Median Filter
img_median = cv2.medianBlur(img_noise, kernel_size)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))


axes[0, 0].imshow(img_noise)
axes[0, 0].set_title('Noisy Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_avg)
axes[0, 1].set_title('Moving Average Filter')
axes[0, 1].axis('off')

axes[1, 0].imshow(img_gaussian)
axes[1, 0].set_title('Gaussian Filter')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_median)
axes[1, 1].set_title('Median Filter')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
#%% Sharpening filter

# Laplacian Filter
laplacian = cv2.Laplacian(img_avg, cv2.CV_64F)
laplacian_8u = cv2.convertScaleAbs(laplacian)
img_laplacian = cv2.add(img_avg, laplacian_8u)

# Sobel Filter
sobel_x = cv2.Sobel(img_avg, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_avg, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sobel_8u = cv2.convertScaleAbs(sobel)
img_sobel = cv2.add(img_avg, sobel_8u)

#%% Display Results

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_avg)
axes[0].set_title('Average Image')
axes[0].axis('off')

axes[1].imshow(img_laplacian)
axes[1].set_title('Laplacian Filter')
axes[1].axis('off')

axes[2].imshow(img_sobel)
axes[2].set_title('Sobel Filter')
axes[2].axis('off')

plt.tight_layout()
plt.show()

#%% PA S%P Filter

flag_mask = np.where(np.logical_or(img==0, img==255), 0, 1)
