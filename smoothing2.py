'''
Python 3.9.16v

date: 2023-04-04, Tues
author: You Ho Yeong, Lee Ga Ram

### Assignment 1 ###
'''

#%% Import Library

import numpy as np
import cv2
import matplotlib.pyplot as plt


#%% Salt and Pepper noise
def spnoise(image, prob): #image : 내가 쓸 데이터, prob : salt and pepper의 비율
    output = image.copy() #원데이터가 영향 받지 않게 copy를 output에 넣어줌
    if len(image.shape) == 2: #image가 흑백인지 아닌지?
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

#%% Read Image

img = cv2.imread(r'/Users/hoyeong/Desktop/python/PBML/Image/Kids_park.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap='gray')
ax.set_title('Original Gray Image')
plt.tight_layout()
plt.show()

img_noise = spnoise(img, 0.3)

plt.hist(img.ravel(), 256, [0,256])
plt.show()




#%% Smoothing Filtering

for i in range(2,10):
    img=cv2.imread(r'/Users/hoyeong/Desktop/python/PBML/Image/Kids_park.jpg')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = spnoise(img,i*0.1)
    cv2.imwrite('/Users/hoyeong/Desktop/python/PBML/Image/sp0{}.jpg'.format(i),result)


sp=['sp02.jpg','sp03.jpg','sp04.jpg','sp05.jpg','sp06.jpg','sp07.jpg','sp08.jpg','sp09.jpg']
i=[2,3,4,5,6,7,8,9]

for i in [9]:
    img_noise = spnoise(img, i * 0.1)
    # Apply moving average filter
    img_moving_avg = cv2.blur(img_noise, (3, 3))

    # Apply median filter
    img_median = cv2.medianBlur(img_noise, 3)

    # Apply Gaussian filter
    img_gaussian = cv2.GaussianBlur(img_noise, (3, 3), 0)

    # Apply bilateral filter
    img_bilateral = cv2.bilateralFilter(img_noise, 9, 75, 75)

    cv2.imwrite('/Users/hoyeong/Desktop/python/PBML/Image/spma_{}0.jpg'.format(i), img_moving_avg)
    cv2.imwrite('/Users/hoyeong/Desktop/python/PBML/Image/spmd_{}0.jpg'.format(i), img_median)
    cv2.imwrite('/Users/hoyeong/Desktop/python/PBML/Image/spga_{}0.jpg'.format(i), img_gaussian)
    cv2.imwrite('/Users/hoyeong/Desktop/python/PBML/Image/spbi_{}0.jpg'.format(i), img_bilateral)
    