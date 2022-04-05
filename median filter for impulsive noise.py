from pickle import NONE
from statistics import median
from tkinter import N
from itertools import chain
import numpy as np
import cv2
import os
import random
import math


# Global variable
input_path = '/Users/yiyun/Documents/Graduate_Study/ME640/project/input/image.pbm'
output_path = '/Users/yiyun/Documents/Graduate_Study/ME640/project/output/'


# Add salt-and pepper noise in an image
def add_noise(img):
    # Getting the dimensions of the image
    row , col = img.shape

    # Randomly pick some pixels in the image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):      
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)    
        # Color that pixel to white
        img[y_coord][x_coord] = 255
         
    # Randomly pick some pixels in the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):     
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)        
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)        
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img
 

# Store the latest image into output path
def output_image(display_name, save_name, image):
    #cv2.imshow(display_name, image)
    cv2.imwrite(output_path + save_name, image)
    print("Image %s%s is saved." % (output_path, save_name))


# Original Median filter 
# Do not depend on the latest value
def median_filter(data, filter_size):
    # Initialization
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    height, width = data.shape[:2]
    N = filter_size // 2

    # Slide window to calculate median value of each pixel
    for i in range(N, height - N):
        for j in range(N, width - N):
            filter_area = data[i - N: i + N + 1, j - N: j + N + 1]
            data_final[i][j] = np.median(filter_area)
    '''
    # Do not include boundary
    for i in range(height):
        for j in range(width):
            if data_final[i][j] == 0:
                data_final[i][j] = data[i][j] 
    '''      
    return data_final


#================================Noise detection=====================================#
def noise_detection(data):
    # Initialize a noise candidate position 2D array
    ncandidate_pos = []
    height, width = data.shape[:2]
    binary_image = np.zeros((len(data),len(data[0])))

    # Use 3*3 window to select noise pixel candidates
    for i in range(1, height - 1):
        for j in range(1, width -1):
            window3 = data[i - 1: i + 2, j - 1: j + 2]
            r1 = window3.min()
            r9 = window3.max()
            if data[i][j] == r1 or data[i][j] == r9:
                ncandidate_pos.append([i,j])
    
    # Use 11*11 window to do the final selection
    for row in range(len(ncandidate_pos)):
        center_row = ncandidate_pos[row][0]
        center_col = ncandidate_pos[row][1]
        # If the position is near the boundary, ensure the window will not out of bound
        min_win11_row = 0
        max_win11_row = 0
        min_win11_col = 0
        max_win11_col = 0
        if center_row >= 5:
            min_win11_row = center_row - 5
        else:
            min_win11_row = 0
        if center_col >= 5:
            min_win11_col = center_col - 5
        else:
            min_win11_col = 0
        if (center_row + 5) < height:
            max_win11_row = center_row + 5
        else:
            max_win11_row = height - 1
        if (center_col + 5) < width:
            max_win11_col = center_col + 5
        else:
            max_win11_col = width - 1
        
        window11 = data[min_win11_row: max_win11_row + 1, min_win11_col: max_win11_col + 1]
        # Sort the window
        R = np.sort(window11, axis=None)
        # Compute distance vector
        distance = np.diff(R)
        # Find four largest distance and their indices
        ind = np.argpartition(distance, -4)[-4:]
        p = ind.min()
        t = ind.max()
        w_min = R[p + 1]
        if w_min == 0:
            w_min = R[p + 1] + 1
        w_max = R[t]

        # Judge the final noise pixel
        # 1 represent noise and 0 represent noise-free pixel
        if data[center_row][center_col] < w_min or data[center_row][center_col] > w_max:
            binary_image[center_row][center_col] = 1

    return binary_image


#================================Noise Removal=====================================#

# Count number of noise pixels in a certain size of window
def nop(certain_area): 
    nop = np.count_nonzero(certain_area)
    return nop


# Generate Normalized Gaussian Surface
def Gaussian_surface(size):
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    dst = np.sqrt(x*x+y*y)
    # Initializing sigma and muu
    sigma = 1
    muu = 0.000
    # Generate Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    # Normalize Gaussian array
    gauss_list = gauss.tolist()
    normalize_gauss = [[1 for x in range(size)] for i in range(size)]
    for i in range(size-1):
        if gauss_list[i][0] <= gauss_list[i+1][0]:
            normalize_gauss[i+1][0] = normalize_gauss[i][0] + 1
        else:
            normalize_gauss[i+1][0] = normalize_gauss[i][0] - 1
    for i in range(size):
        for j in range(size-1):
            if gauss_list[i][j] <= gauss_list[i][j+1]:
                normalize_gauss[i][j+1] = normalize_gauss[i][j] + 1
            else:
                normalize_gauss[i][j+1] = normalize_gauss[i][j] - 1
    return normalize_gauss


# Dynamically Weighted Median Filter Algorithm
def DWMF(part_img, binary, size):
    # Initialiazation
    new_value = 0
    store_list = []
    # Assume ths size of img, binary and guassian have the same size
    rev_binary_image = 1 - binary
    gaussian = Gaussian_surface(size)
    # Calculate Wweight
    W_weight = np.multiply(rev_binary_image, gaussian)
    height, width = W_weight.shape[:2]
    # Weighted Median Filter
    for i in range(height):
        for j in range(width):
            if W_weight[i][j] != 0:
                freq = W_weight[i][j].astype(int)
                current_new_list = [part_img[i][j].astype(int)] * freq
                store_list.extend(current_new_list)
    store_list.sort()
    new_value = median(store_list)
    return new_value

# ADWMF
def ADWMF(noise_img, binary_image):
    # Initialization
    final_image = np.zeros((len(noise_img),len(noise_img[0])))
    height, width = noise_img.shape[:2]
    # Do not consider the boundry
    for i in range(3, height - 3):
        for j in range(3, width - 3):
            # Firstly we assume that the slide window size is 3x3
            N = 1
            part_bimg =binary_image[i - N: i + N + 1, j - N: j + N + 1]
            num = nop(part_bimg)
            if num < 4:
                part_img = noise_img[i - N: i + N + 1, j - N: j + N + 1] 
                final_image[i][j] = DWMF(part_img, part_bimg, 3)
            else:
                N = 2
                part_bimg =binary_image[i - N: i + N + 1, j - N: j + N + 1]
                num = nop(part_bimg)
                if num < 13:
                    part_img = noise_img[i - N: i + N + 1, j - N: j + N + 1]
                    final_image[i][j] = DWMF(part_img, part_bimg, 5)
                else:
                    N = 3
                    part_bimg =binary_image[i - N: i + N + 1, j - N: j + N + 1]
                    part_img = noise_img[i - N: i + N + 1, j - N: j + N + 1]
                    final_image[i][j] = DWMF(part_img, part_bimg, 7)
    
    # Deal with the boundary
    merged_i = chain(range(0, 3), range(height - 3, height))
    for i in merged_i:
        for j in range(width):
            N = 1
            a = i - N
            b = i + N + 1
            c = j - N
            d = j + N + 1
            if a < 0:
                a = 0
            if c < 0:
                c = 0
            if b > height:
                b = height
            if d > width:
                d = width
            part_img = noise_img[a: b, c: d] 
            final_image[i][j] = np.median(part_img)

    for i in range(3, height - 3):
        for j in range(0, 3):
            N = 1
            a = i - N
            b = i + N + 1
            c = j - N
            d = j + N + 1
            if a < 0:
                a = 0
            if c < 0:
                c = 0
            if b > height:
                b = height
            if d > width:
                d = width
            part_img = noise_img[a: b, c: d] 
            final_image[i][j] = np.median(part_img)
     
    for i in range(3, height - 3):
        for j in range(width - 3, width):
            N = 1
            a = i - N
            b = i + N + 1
            c = j - N
            d = j + N + 1
            if a < 0:
                a = 0
            if c < 0:
                c = 0
            if b > height:
                b = height
            if d > width:
                d = width
            part_img = noise_img[a: b, c: d] 
            final_image[i][j] = np.median(part_img)
    return final_image


#================================Evaluation=====================================#
# Peak Signal to Noise Ratio
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def main():
    # Image as an input
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Read original image and make them as a grayscale image
    img = cv2.imread(input_path, 0)
    output_image('original_image', 'original_image.jpg', img)
    print("Completed the original image.")

    # Add salt-and-pepper on original image
    noise_add_img = add_noise(img)
    output_image('noise_image', 'noise_image.jpg', noise_add_img)
    print("Completed the noise image.")

    # Noise removal by using normal median filter
    # Slide window size is 3x3
    original_result = median_filter(noise_add_img, 3)
    # Store the latest image into output dictinary
    output_image('MF', 'MF.jpg', original_result)
    print("Completed the image_MF.")

    
    # Noise removal depend on the detection result
    # Parameters initialization
    result_final = []
    result_final = np.zeros((len(noise_add_img),len(noise_add_img[0])))

    # Noise detection
    binary_image = noise_detection(noise_add_img)
    
    # Noise removal by using ADWMF
    result_final = ADWMF(noise_add_img, binary_image)
    
    output_image('ADWMF', 'ADWMF.jpg', result_final)
    print("Completed the image_ADWMF.")

    # Evaluate the image quality (PSNR)
    psnr_common = psnr(img, original_result)
    print("PSNR of image by using original median filter: ", psnr_common)
    psnr_AD = psnr(img, result_final)
    print("PSNR of image by using ADWMF: ", psnr_AD)

    # Destroy all the images on any key press.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()