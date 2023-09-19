import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def adjust_colorness(image, percentage):
    """
    Adjust the color intensity of an image.
    Args:
    - image: Input image which can be grayscale or colored.
    - percentage: Percentage of color intensity desired. 0% means grayscale, 
                  100% means original color, and any value in between scales the color intensity.
    Returns:
    - Image with adjusted color intensity.
    """
    if len(image.shape) == 2 or percentage == 0:
        return np.mean(image, axis=-1)
    
    grayscale = np.mean(image, axis=-1)
    color_adjusted = grayscale[:, :, np.newaxis] + percentage / 100.0 * (image - grayscale[:, :, np.newaxis])
    return np.clip(color_adjusted, 0, 1)

def create_hybrid_image_v4(im1, im2, sigma1, sigma2):
    if len(im1.shape) == 2:
        im1 = np.stack([im1, im1, im1], axis=-1)
    if len(im2.shape) == 2:
        im2 = np.stack([im2, im2, im2], axis=-1)

    # Low-pass filter for im1
    low_passed_im1 = gaussian_filter(im1, sigma=(sigma1, sigma1, 0), mode='nearest')
    
    # High-pass filter for im2
    low_passed_im2 = gaussian_filter(im2, sigma=(sigma2, sigma2, 0), mode='nearest')
    high_passed_im2 = im2 - low_passed_im2
    
    # Plotting
    # plt.figure(figsize=(10,5))  # Adjust the figure size if needed

    # plt.subplot(1, 2, 1)
    # plt.imshow(low_passed_im1)
    # plt.title("Low-passed Image 1")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(high_passed_im2)
    # plt.title("High-passed Image 2")
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    
    # Combine to form hybrid image
    hybrid = np.clip(low_passed_im1 + high_passed_im2, 0, 1)
    return hybrid


def create_and_display_hybrids_v5(im1, im2, sigma1, sigma2, colorness1=100, colorness2=100):
    """
    Create and display hybrid images for different combinations of input images.
    """
    # Adjust colorness
    im1_colored = adjust_colorness(im1, colorness1)
    im2_colored = adjust_colorness(im2, colorness2)

    # Create hybrid images for different combinations
    hybrids = {
        'Low f B&W to High f B&W': create_hybrid_image_v4(adjust_colorness(im1_colored, 0), adjust_colorness(im2_colored, 0), sigma1, sigma2),
        'Low f B&W to High f Colored (%d%%)' % colorness2: create_hybrid_image_v4(adjust_colorness(im1_colored, 0), im2_colored, sigma1, sigma2),
        'Low f Colored (%d%%) to High f B&W' % colorness1: create_hybrid_image_v4(im1_colored, adjust_colorness(im2_colored, 0), sigma1, sigma2),
        'Low f Colored (%d%%) to High f Colored (%d%%)' % (colorness1, colorness2): create_hybrid_image_v4(im1_colored, im2_colored, sigma1, sigma2)
    }
    
    # Display the hybrid images
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for idx, (title, hybrid) in enumerate(hybrids.items()):
        ax[idx].imshow(hybrid)
        ax[idx].axis('off')
        ax[idx].set_title(title)
    
    plt.tight_layout()
    plt.show()


# Load the provided images
im1 = plt.imread('./images/hi2.jpg')/255.
im2 = plt.imread('./images/hiAgain2.jpg')/255.

create_and_display_hybrids_v5(im1, im2, 14, 25, colorness1=10, colorness2=100)
