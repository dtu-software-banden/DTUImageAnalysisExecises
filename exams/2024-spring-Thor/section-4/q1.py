import cv2
import numpy as np

# Load grayscale zebra image
zebra = cv2.imread('Zebra.png', cv2.IMREAD_GRAYSCALE)

# Load binary masks
mask_white = cv2.imread('Zebra_whiteStripes.png', cv2.IMREAD_GRAYSCALE)
mask_black = cv2.imread('Zebra_blackStripes.png', cv2.IMREAD_GRAYSCALE)
mask_area = cv2.imread('Zebra_MASK.png', cv2.IMREAD_GRAYSCALE)

# Convert masks to boolean
white_pixels = zebra[mask_white > 0]
black_pixels = zebra[mask_black > 0]

# Estimate Gaussian parameters
mean_white = np.mean(white_pixels)
std_white = np.std(white_pixels)
mean_black = np.mean(black_pixels)
std_black = np.std(black_pixels)

# Flatten zebra image and mask
zebra_flat = zebra[mask_area > 0]

# Compute Gaussian likelihoods
def gaussian(x, mean, std):
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(- ((x - mean) ** 2) / (2 * std ** 2))

likelihood_white = gaussian(zebra_flat, mean_white, std_white)
likelihood_black = gaussian(zebra_flat, mean_black, std_black)

# Classify
classified_white = likelihood_white > likelihood_black

# Count white stripe pixels
white_pixel_count = np.sum(classified_white)
print(f"Number of pixels classified as white stripe: {white_pixel_count}")


# Get original pixel values classified as black
classified_black = ~classified_white  # Invert white classification
black_pixels_classified = zebra_flat[classified_black]

# Get intensity range
min_black = np.min(black_pixels_classified)
max_black = np.max(black_pixels_classified)

print(f"Class range for black stripes: {min_black} to {max_black}")
print(f"White Stripe Class Gaussian Parameters:")
print(f"Mean: {mean_white:.2f}")
print(f"Standard Deviation: {std_white:.2f}")
