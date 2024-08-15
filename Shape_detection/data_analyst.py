import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original and drawing images
original = cv2.imread('D:/WWU/M8 - Master Thesis/Project/Code/Images/2.jpg')
drawing = cv2.imread('D:/WWU/M8 - Master Thesis/Project/Code/Images/two.jpg')

# Convert to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

# Resize the drawing to match the original image size
drawing_gray_resized = cv2.resize(drawing_gray, (original_gray.shape[1], original_gray.shape[0]))

# Calculate the absolute difference between the original and the drawing
difference = cv2.absdiff(original_gray, drawing_gray_resized)

# Apply a binary threshold to highlight significant differences
_, thresholded_diff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

# Display the original, drawing, and difference images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(original_gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Drawing")
plt.imshow(drawing_gray_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(thresholded_diff, cmap='gray')
plt.axis('off')

plt.show()

# Optionally, save the difference image
cv2.imwrite('difference_image.png', thresholded_diff)
