import numpy as np
import cv2
import os
import random

# Global variable
input_path = '/Users/ZhenhaoYou/Desktop/python/peppernosie.jpg'
output_path = '/Users/ZhenhaoYou/Desktop/python/outputs/'

innermatrix = []
exp = []
for i in range(5):
    exp.append([])
    for j in range(5):
        exp[i].append(random.randint(0,100))
    
print(np.array(exp))
# Simple Example
# exp = [[4,2,1,8,7], [6,3,7,5,4], [1,3,4,8,2], [6,4,3,5,7], [3,5,5,8,9]]


# Store the latest image into output path
def output_image(display_name, save_name, image):
    #cv2.imshow(display_name, image)
    cv2.imwrite(output_path + save_name, image)
    print("Image %s%s is saved." % (output_path, save_name))


# Median filter 
# Do not depend on the latest value
def median_filter(data, filter_size):
    # Initialization
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    
    '''
    # Add the boundary value (same as the side row & column)
    # This is only suitable for 3*3 slide window, how to include boundary for dynamical window size?
    data = np.insert(data, 0, data[0, :], 0)
    data = np.insert(data, -1, data[-1, :], 0)
    start_col = data[:, 0]
    end_col = data[:, -1]
    start_col.shape = (start_col.size,1)
    end_col.shape = (end_col.size, 1)
    data = np.hstack((start_col, data))
    data = np.hstack((data, end_col))
    '''

    height, width = data.shape[:2]
    N = filter_size // 2

    # Slide window to calculate median value of each pixel
    for i in range(N, height - N):
        for j in range(N, width - N):
            filter_area = data[i - N: i + N + 1, j - N: j + N + 1]
            data_final[i][j] = np.median(filter_area)
    # Do not include boundary
    for i in range(height):
        for j in range(width):
            if data_final[i][j] == 0:
                data_final[i][j] = data[i][j]           
    return data_final


# Noise detection
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
        min_win11_row=0
        max_win11_row=0 
        min_win11_col=0
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
        '''
        key_list = []
        value_list = []
        for a in ind:
            key_list.append(a)
            value_list.append(distance[a])
        zip_iterator = zip(key_list, value_list)
        dictionary = dict(zip_iterator)
        '''
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


def main():
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Read original image
    img = cv2.imread(input_path, 0)
    cv2.imshow('origin_image', img)
    # Noise removal
    result = median_filter(img, 3)
    # Store the latest image into output dictinary
    output_image('new_image1', 'new_image1.jpg', result)
    print("Completed all of the images.")

    # Destroy all the images on any key press.
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    # Simple example
    arr = np.array(exp)   
    result = noise_detection(arr)
    print(result)
    

if __name__ == "__main__":
    main()