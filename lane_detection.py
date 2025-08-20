# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('lane.jpeg')  # Make sure 'lane.jpeg' is in the same directory or give full path
if image is None:
    raise FileNotFoundError("Image 'lane.jpeg' not found!")

# Get image dimensions
height, width = image.shape[:2]

# Make a copy of the image
img_copy = np.copy(image)

# -------------------------------
# Define region of interest mask
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # For a single channel grayscale image
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Define vertices for region of interest
region_of_interest_vertices = np.array([[
    (0, height),
    (width // 2, height // 2),
    (width, height)
]], dtype=np.int32)

# Convert to grayscale and apply Canny edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 200)

# Crop the image to region of interest
cropped_image = region_of_interest(canny, region_of_interest_vertices)

# Show cropped Canny image
plt.figure(figsize=(10, 5))
plt.imshow(cropped_image, cmap='gray')
plt.title("Cropped Canny Edge Detection")
plt.axis('off')
plt.show()

# -------------------------------
# Laplacian Edge Detection
blurred_img = cv2.GaussianBlur(gray, (3, 3), 0)
laplacian = cv2.Laplacian(blurred_img, cv2.CV_64F)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(blurred_img, cmap='gray')
plt.title('Blurred Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.axis('off')
plt.show()

# -------------------------------
# Sobel Edge Detection
sobel_y = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
])

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Apply Sobel filters
sobel_filtered_y = cv2.filter2D(gray, -1, sobel_y)
sobel_filtered_x = cv2.filter2D(gray, -1, sobel_x)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_filtered_x, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_filtered_y, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')
plt.show()

# -------------------------------
# Canny Edge Detection (Again for comparison)
edges = cv2.Canny(blurred_img, 250, 250)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(blurred_img, cmap='gray')
plt.title('Blurred Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()

