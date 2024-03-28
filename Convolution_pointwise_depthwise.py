import numpy as np
from scipy.signal import convolve2d
from PIL import Image


class ImageConvolver:
    def __init__(self, image_path):
        """Initialize with the path to the image to be convolved."""
        self.image_path = image_path
        self.image = self._load_image()

    def _load_image(self):
        """Load an image from the path and convert it to a numpy array."""
        with Image.open(self.image_path) as img:
            return np.asarray(img)

    def _save_image(self, array, output_path):
        """Save a numpy array as an image."""
        Image.fromarray(np.uint8(array)).save(output_path)

    def apply_depthwise_convolution(self, kernel):
        """Apply depthwise convolution using the specified kernel."""
        if self.image.ndim == 3:
            convolved = np.stack([
                convolve2d(self.image[:, :, i], kernel, mode='same', boundary='wrap')
                for i in range(self.image.shape[2])
            ], axis=2)
        else:
            convolved = convolve2d(self.image, kernel, mode='same', boundary='wrap')
        return convolved

    def perform_convolution(self, kernel, pointwise_filter=None, mode='depthwise'):
        """
        Perform convolution based on the specified mode. This method orchestrates
        the application of depthwise and pointwise convolutions.
        """
        if mode == 'depthwise':
            convolved = self.apply_depthwise_convolution(kernel)
        elif mode == 'pointwise':
            if pointwise_filter is not None:
                convolved = self.apply_pointwise_convolution(pointwise_filter)
            else:
                raise ValueError("Pointwise filter must be provided for pointwise convolution.")
        else:
            raise ValueError("Unsupported mode. Choose either 'depthwise' or 'pointwise'.")

        # Ensure the convolved image retains a channel dimension if it's a single channel
        if convolved.ndim == 2:
            convolved = np.expand_dims(convolved, axis=-1)
        return convolved

    def apply_pointwise_convolution(self, pointwise_filter):
        """
        Apply pointwise convolution using the specified filter. This method now
        accommodates images with any number of channels, including single-channel.
        """
        if self.image.ndim == 3:  # Image has multiple channels
            depth = self.image.shape[2]
            if pointwise_filter.shape[0] == depth:
                # Properly preparing the pointwise filter for convolution
                pointwise_filter = pointwise_filter.reshape((1, 1, depth, pointwise_filter.shape[1]))
                # Applying the convolution
                convolved = np.einsum('ijkl,imjk->iml', pointwise_filter, self.image)
                return convolved
            else:
                raise ValueError(f"Filter depth {pointwise_filter.shape[0]} does not match image depth {depth}.")
        else:
            raise ValueError("Image must have multiple channels for pointwise convolution.")

if __name__=="__main__":
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Kernel for depthwise convolution
    pointwise_filter = np.array([[1, 0, 1]])  # Filter for pointwise convolution

    convolver = ImageConvolver('bird.png')

    # Perform depthwise convolution
    depthwise_result = convolver.perform_convolution(kernel, mode='depthwise')

    # Perform pointwise convolution on the depthwise convolved image
    convolver.image = depthwise_result  # Update the internal image to the depthwise convolved one
    pointwise_result = convolver.perform_convolution(None, pointwise_filter=pointwise_filter, mode='pointwise')

    # Save results
    convolver._save_image(depthwise_result, 'depthwise_convoluted_image.png')
    convolver._save_image(pointwise_result, 'pointwise_convoluted_image.png')
