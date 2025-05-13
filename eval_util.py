
import numpy as np
import cv2 as cv
import os
import copy
from scipy.ndimage import label,distance_transform_edt


def min_area_check(image, verbose=0):
    """
    Calculate the area of isolated shapes in a binary image.
    Parameters:
    image (numpy.ndarray): A binary image (2D array) where the pixels inside the shapes have a value of 1
                           and the pixels outside have a value of 0.
    Returns:
    list of int: A list containing the area (number of pixels) of each isolated shape in the image.
    """
    # Label the connected components (shapes) in the binary image
    labeled_array, num_features = label(image)
    # Calculate the area of each isolated shape
    areas = []
    for i in range(1, num_features + 1):
        area = np.sum(labeled_array == i)
        areas.append(area)
    areas=np.array(areas)
    if verbose:
        return areas
    else:
        return np.min(areas)
    
def min_dist_check(image, verbose=0): 
    """
    Calculate the minimal distance in terms of pixels between any two isolated shapes in a binary image.

    Parameters:
    image (numpy.ndarray): A binary image (2D array) where the pixels inside the shapes have a value of 1
                           and the pixels outside have a value of 0.

    Returns:
    int: The minimal distance between any two isolated shapes.
    """
    # Label the connected components (shapes) in the binary image
    labeled_array, num_features = label(image)

    # Initialize minimum distance as a large number
    min_distance = float('inf')

    # Loop through each shape
    for i in range(1, num_features + 1):
        # Create a binary mask for the current shape
        shape_mask = (labeled_array == i)

        # Compute the distance transform of the inverted mask (i.e., distance to the nearest zero)
        dist_transform = distance_transform_edt(~shape_mask)

        # Loop through each other shape and find the minimum distance
        for j in range(1, num_features + 1):
            if i != j:
                other_shape_mask = (labeled_array == j)
                min_distance = min(min_distance, np.min(dist_transform[other_shape_mask]))

    return int(min_distance)


if __name__=="__main__":
    import cv2
    path = "./benchmarks/M1_test2/M1_test2.png.mask.png"
    maski=cv2.imread(path,-1)/255.0
    print(min_dist_check(maski))
    print(min_area_check(maski))