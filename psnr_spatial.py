'''
Python 3.9.16v

date: 2023-04-16, Sun
author: You Ho Yeong, Lee Ga Ram

### Calculating Peark Singal to Noise Ratio(PSNR) ###
'''
#%% Import Library
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%% Calculate PSNR of Smoothing Methods
"""
For Spatial Filtering Methods
Moving Average, Median, Gaussian, Bilateral, Proposed Algorithm (TOTAL == 5)
"""
img = cv2.imread('/Users/hoyeong/Desktop/python/PBML/Image/Kids_park.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the filters and their labels
filters = ['PA', 'spbi', 'spga', 'spma', 'spmd']
labels = ['Proposed Algorithm', 'Bilateral', 'Gaussian', 'Moving Average', 'Median']
markers = ['o', 's', 'D', '*', '^']

# Define the range of noise densities
densities = range(20, 100, 10)

# Initialize a dictionary to store the PSNR values for each method
psnr_values = {filter_label: [] for filter_label in labels}

for filter_short, filter_label, marker in zip(filters, labels, markers):
    for density in densities:
        # Load the filtered image
        img_fil = cv2.imread('/Users/hoyeong/Desktop/python/PBML/Image/{}_{}.jpg'.format(filter_short, density),
                                                                                          cv2.IMREAD_GRAYSCALE)
        if img_fil is None:
            print('Failed to load the image for filter {} and density {}'.format(filter_short, density))
            continue

        if img.shape != img_fil.shape:
            print('The size of the original image and filtered image do not match for filter {} and density {}'.format(filter_short, density))
            continue
        # Calculate the PSNR
        psnr = cv2.PSNR(img, img_fil)
        psnr_values[filter_label].append(psnr)
        print('PSNR for {} and density {}: {:.2f} dB'.format(filter_label, density, psnr))
        
# Plot the results
for filter_label, marker in zip(labels, markers):
    plt.plot(densities, psnr_values[filter_label], marker=marker, label=filter_label)

plt.xlabel('Noise Density: P[%]')
plt.ylabel('PSNR[dB]')
plt.title('PSNR')
plt.legend()
plt.grid(True)
plt.show()
