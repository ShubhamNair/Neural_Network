import numpy as np
import matplotlib.pyplot as plt

def convolve2d(image, kernel):
    """
    Perform a 2D convolution operation without padding and striding.
    Args:
        image: 2D numpy array representing the image.
        kernel: 2D numpy array representing the filter.
    Returns:
        2D numpy array representing the convolved image.
    """
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    output_height = i_height - k_height + 1
    output_width = i_width - k_width + 1
    output = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(image[y:y + k_height, x:x + k_width] * kernel)
    return output

# Example: Convolution operation without padding and striding

# Original image (6x6)
original_image = np.array([
    [1, 3, 2, 1, 0, 2],
    [0, 2, 1, 0, 1, 1],
    [1, 1, 0, 2, 3, 1],
    [2, 2, 1, 1, 1, 0],
    [0, 1, 3, 1, 2, 2],
    [1, 0, 1, 3, 2, 0]
])

# Filter kernel (3x3)
filter_kernel = np.array([
    [2, 0, 1],
    [1, 1, 0],
    [0, 2, 1]
])

# Perform convolution
convolved_image = convolve2d(original_image, filter_kernel)

# Display the results
print("Original Image:\n", original_image)
print("\nFilter Kernel:\n", filter_kernel)
print("\nConvolved Image:\n", convolved_image)

# Plot the original image, filter kernel, and convolved image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray', interpolation='nearest')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(filter_kernel, cmap='gray', interpolation='nearest')
plt.title('Filter Kernel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(convolved_image, cmap='gray', interpolation='nearest')
plt.title('Convolved Image')
plt.axis('off')

plt.show()
