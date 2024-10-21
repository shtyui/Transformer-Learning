# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:33:17 2023

@author: s205
"""

import numpy as np

# Define the input image as a NumPy array
input_image = np.array([
    [7, 5, 1, 3, 4],
    [2, 6, 2, 4, 3],
    [3, 7, 4, 3, 7],
    [6, 4, 3, 7, 5],
    [1, 2, 5, 6, 4]
], dtype=int)

# Initialize an empty array for the integral image
integral_image = np.zeros_like(input_image, dtype=int)

# Calculate the first element of the integral image
integral_image[0, 0] = input_image[0, 0]

# Calculate the values for the first row of the integral image
for j in range(1, input_image.shape[1]):
    integral_image[0, j] = integral_image[0, j - 1] + input_image[0, j]

# Calculate the values for the first column of the integral image
for i in range(1, input_image.shape[0]):
    integral_image[i, 0] = integral_image[i - 1, 0] + input_image[i, 0]

# Fill in the remaining values in the integral image
for i in range(1, input_image.shape[0]):
    for j in range(1, input_image.shape[1]):
        integral_image[i, j] = (integral_image[i - 1, j] +
                                integral_image[i, j - 1] -
                                integral_image[i - 1, j - 1] +
                                input_image[i, j])

# Print the integral image
print("Integral Image:")
print(integral_image)
