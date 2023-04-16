'''
Python 3.9.16v

date: 2023-04-16, Sun
author: You Ho Yeong, Lee Ga Ram

### Calculating Rice Distance for Edges ###
'''
#%% Import Library
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%% Processing

def detect_edges(image, low_threshold=100, high_threshold=200):
    '''
    Get Edges using Canny Filter
    '''
    image_8bit = cv2.convertScaleAbs(image)
    return cv2.Canny(image_8bit, low_threshold, high_threshold)


def rdist(edge_image):
    row, col = edge_image.shape
    rise_distances = []
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if edge_image[i, j] == 255:
                local_rise_distance = np.sqrt(((np.int16(edge_image[i + 1, j]) - np.int16(edge_image[i - 1, j]))**2)
                                            + ((np.int16(edge_image[i, j + 1]) - np.int16(edge_image[i, j - 1]))**2))
                rise_distances.append(local_rise_distance)
    
    return np.mean(rise_distances) if rise_distances else 0


def rmse(predictions, targets):
    """
    Compute Root Mean Square Error (RMSE) between predictions and targets.

    Parameters:
    predictions (numpy.array or list): Output data from filter 1
    targets (numpy.array or list): Output data from filter 2

    Returns:
    float: The RMSE value
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    if predictions.shape != targets.shape:
        raise ValueError("The shape of predictions and targets must be the same.")

    return np.sqrt(np.mean((predictions - targets) ** 2))


# Load Original Image
img = cv2.imread(r'/Users/hoyeong/Desktop/python/PBML/Image/Kids_park.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Laplacian, Sobel, USM, and Canny filters to the original grayscale image
sigma = [0, 0.1, 0.15, 0.2, 0.25,0.3]
# rdlap, rdsob, rdusm, rdcan = [], [], [], []

for i in sigma:
    gaus_img=cv2.GaussianBlur(img, (9, 9), i)

    # Apply Gaussian blur for SOBEL
    sobelx_img = cv2.Sobel(np.uint8(gaus_img), cv2.CV_64F, 1, 0, ksize=5)
    sobely_img = cv2.Sobel(np.uint8(gaus_img), cv2.CV_64F, 0, 1, ksize=5)
    sobelx_abs = cv2.convertScaleAbs(sobelx_img)
    sobely_abs = cv2.convertScaleAbs(sobely_img)

    gaus_sob = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

    # gaussian_img = cv2.GaussianBlur(gaus_img, (5, 5), 0)
    # gaus_usm = cv2.addWeighted(gaus_img, 1.5, gaussian_img, -0.5, 0)

    # gaus_can = cv2.Canny(gaus_img, 100, 200)
    # gaus_lap = cv2.Laplacian(gaus_img, cv2.CV_64F)

    error = rmse(img, gaus_sob)
    print("RMSE {}: {}".format(i,error))