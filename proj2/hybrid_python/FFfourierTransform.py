import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

# Load the provided images
im1 = plt.imread('./images/hi2.jpg')/255.
im2 = plt.imread('./images/hiAgain2.jpg')/255.



# Check if the images are grayscale or colored
if len(im1.shape) > 2:
    im1_gray = np.mean(im1, axis=-1)
else:
    im1_gray = im1

if len(im2.shape) > 2:
    im2_gray = np.mean(im2, axis=-1)
else:
    im2_gray = im2

# # Display the two images
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(im1_gray, cmap='gray')
# ax[0].axis('off')
# ax[0].set_title('Image 1')
# ax[1].imshow(im2_gray, cmap='gray')
# ax[1].axis('off')
# ax[1].set_title('Image 2')
# plt.tight_layout()
# plt.show()


def create_hybrid_image(im1, im2, sigma1, sigma2):
    # Low-pass filter for im1
    low_passed_im1 = gaussian_filter(im1, sigma=sigma1)
    
    # High-pass filter for im2
    low_passed_im2 = gaussian_filter(im2, sigma=sigma2)
    high_passed_im2 = im2 - low_passed_im2
    
    # Combine to form hybrid image
    hybrid = np.clip(low_passed_im1 + high_passed_im2, 0, 1)
    return hybrid

# Set arbitrary values for sigma
sigma1 = 14
sigma2 = 25

# Create hybrid image
hybrid_image = create_hybrid_image(im1_gray, im2_gray, sigma1, sigma2)

# Display the hybrid image
plt.figure(figsize=(6,6))
plt.imshow(hybrid_image, cmap='gray')
plt.axis('off')
plt.title('Hybrid Image')
plt.show()


def compute_fourier_transform(img):
    """Compute the Fourier transform and shift the zero frequency component to the center."""
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))) + 1)

# Compute Fourier transforms
im1_ft = compute_fourier_transform(im1_gray)
im2_ft = compute_fourier_transform(im2_gray)
low_passed_im1_ft = compute_fourier_transform(gaussian_filter(im1_gray, sigma=sigma1))
high_passed_im2_ft = compute_fourier_transform(im2_gray - gaussian_filter(im2_gray, sigma=sigma2))
hybrid_ft = compute_fourier_transform(hybrid_image)

# Display frequency analysis results
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Input images
ax[0, 0].imshow(im1_ft, cmap='hot')
ax[0, 0].axis('off')
ax[0, 0].set_title('FT of Image 1')
ax[0, 1].imshow(im2_ft, cmap='hot')
ax[0, 1].axis('off')
ax[0, 1].set_title('FT of Image 2')

# Filtered images
ax[1, 0].imshow(low_passed_im1_ft, cmap='hot')
ax[1, 0].axis('off')
ax[1, 0].set_title('FT of Low-passed Image 1')
ax[1, 1].imshow(high_passed_im2_ft, cmap='hot')
ax[1, 1].axis('off')
ax[1, 1].set_title('FT of High-passed Image 2')

# Hybrid image
ax[1, 2].imshow(hybrid_ft, cmap='hot')
ax[1, 2].axis('off')
ax[1, 2].set_title('FT of Hybrid Image')

# Hide unused subplot
ax[0, 2].axis('off')

plt.tight_layout()
plt.show()