'''
Python 3.9.16v

date: 2023-04-04, Tues
author: You Ho Yeong, Lee Ga Ram

### Custom Filter using noise Density ###
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

#%% Load file
img = cv2.imread(r'/Users/hoyeong/Desktop/python/PBML/Image/Kids_park.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap='gray')
ax.set_title('Original Gray Image')
plt.tight_layout()
plt.show()

#%% Add noise
img_noise = spnoise(img, 0.8)

#%% Custum Filter

def no_noise_pixels(img):
    return (img != 0) & (img != 255)

def cross_mask(img, i, j):
    north, south, east, west = img[i - 1, j], img[i + 1, j], img[i, j + 1], img[i, j - 1]
    return [direction for direction in [north, south, east, west] if no_noise_pixels(img[i, j])]

def calculate_cross_mask_pixel(img, i, j):
    cross_pixels = cross_mask(img, i, j)
    if len(cross_pixels) >= 2:
        return 0.5 * (max(cross_pixels) + min(cross_pixels))
    elif len(cross_pixels) == 1:
        return cross_pixels[0]
    return None

def calculate_mask_pixel(img, i, j, k, nm2):
    if k % 2 == 1:  # k is odd
        return np.median(nm2)
    elif k % 2 == 0:  # k is Even
        nm2.sort()
        return 0.5 * (nm2[int(k / 2 - 1)] + nm2[int(k / 2)])

def calculate_large_mask_pixel(img, i, j, noise_density, large_mask):
    if noise_density < 0.5:
        return weighted_distance_calculation(img, i, j, large_mask)
    elif noise_density > 0.5:
        return max_non_noise_region(img, i, j, large_mask)
    return None

def weighted_distance_calculation(img, i, j, large_mask):
    f = np.where(np.logical_or(large_mask == 0, large_mask == 255), 0, 1)
    w = np.array([[((1 - p) ** 2 + (1 - q) ** 2) ** 0.5 for q in range(6)] for p in range(6)])
    w = (w + 1) ** 3
    return np.sum(f * w * large_mask) / np.sum(f * w)

def max_non_noise_region(img, i, j, large_mask):
    I, II, III, IV = large_mask[:3, :3], large_mask[:3, 3:], large_mask[3:, :3], large_mask[3:, 3:]
    max_region = max([I, II, III, IV], key=lambda region: np.sum(no_noise_pixels(region)))
    non_noise_count = np.sum(no_noise_pixels(max_region))

    if non_noise_count > 0:
        return calculate_mask_pixel(img, i, j, non_noise_count, [elem for elem in max_region.flatten()])
    return None

def custom_filter(img):
    '''
    Custom Filter to Remove S&P Noise
    Input: Noisy Image
    Output: Filtered Image
    '''
    img = img.astype(np.int64)
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    h, w = img.shape
    output = np.zeros((h, w), dtype=np.int64)

    for i in range(4, h - 4):
        for j in range(4, w - 4):
            if no_noise_pixels(img[i, j]):
                output[i, j] = img[i, j]
            else:
                result = calculate_cross_mask_pixel(img, i, j)
                if result is not None:
                    output[i, j] = result
                else:
                    three_by_three = img[i - 1:i + 2, j - 1:j + 2].flatten()
                    nm2 = [elem for elem in three_by_three if elem not in (0, 255)]
                    k = len(nm2)

                    if k > 0:
                        output[i, j] = calculate_mask_pixel(img, i, j, k, nm2)
                    else:
                        six_by_six = img[i - 1:i + 5, j - 1:j + 5]
                        noise_density = 1 - (np.sum(no_noise_pixels(six_by_six)) / 36)

                        result = calculate_large_mask_pixel(img, i, j, noise_density, six_by_six)
                        if result is not None:
                            output[i, j] = result
                        else:
                            output[i, j] = 0.25 * (output[i - 1, j - 1] + output[i - 1, j]
                                                   + output[i - 1, j + 1] + output[i, j - 1])

    return output[4:-4, 4:-4]
#%% Run algorithm and Compare

filtered_img = custom_filter(img_noise)
median_img = cv2.medianBlur(img_noise, 9)

fig, axes = plt.subplots(figsize=(8, 8))
axes.imshow(img,cmap='gray')
axes.set_title('Original Image')
axes.axis('on')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(figsize=(8, 8))
axes.imshow(img_noise,cmap='gray')
axes.set_title('Salt & Pepper Image')
axes.axis('on')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(figsize=(8, 8))
axes.imshow(median_img,cmap='gray')
axes.set_title('Median Filter')
axes.axis('on')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(figsize=(8, 8))
axes.imshow(filtered_img,cmap='gray')
axes.set_title('PA_{} Filter'.format(0.8))
fig.tight_layout()
axes.axis('on')
fig.savefig('/Users/hoyeong/Desktop/python/PBML/Image/PA_{}.jpg'.format(0.8))

#%% PSNR(Peak Singal to Noise Ratio)

# cv2.PSNR(original, filtered)
filtered_img_uint8 = filtered_img.astype(np.uint8)
psnr_med = cv2.PSNR(img, median_img)
psnr_pa = cv2.PSNR(img, filtered_img_uint8)

print("PSNR_Median value: ", psnr_med)
print("PSNR_PA value: ", psnr_pa)

#%% Save File

p_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for probs in p_list:
    img_noise = spnoise(img, probs)
    pa_img = custom_filter(img_noise)
    print("%.1f Complete" %probs)
    cv2.imwrite('/Users/hoyeong/Desktop/python/PBML/Image/PA_{}.jpg'.format(probs*100),pa_img)
