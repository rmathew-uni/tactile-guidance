import cv2
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

def get_edge_points(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No image found at {image_path}.")
        return []

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detector
    edge_image = cv2.Canny(gray_image, 100, 200)

    # Extract coordinates of non-zero points
    coordinates = [(x, y) for y in range(edge_image.shape[0]) for x in range(edge_image.shape[1]) if edge_image[y, x] > 0]

    return coordinates

def apply_tps_transformation(source_coords, target_coords, source_image_path):
    # Load the source image
    source_image = cv2.imread(source_image_path)
    if source_image is None:
        print(f"Error: Source image not found at {source_image_path}.")
        return None, None

    # Ensure the coordinates have the same length
    if len(source_coords) != len(target_coords):
        print(f"Error: Number of source coordinates ({len(source_coords)}) does not match number of target coordinates ({len(target_coords)}).")
        return None, None

    # Convert coordinates to numpy arrays
    source_points = np.array(source_coords)
    target_points = np.array(target_coords)

    # Create RBF interpolation functions
    try:
        rbf_x = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 0], function='thin_plate')
        rbf_y = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 1], function='thin_plate')
    except Exception as e:
        print(f"Error creating RBF functions: {e}")
        return None, None

    # Create a mesh grid of the image
    rows, cols, _ = source_image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Apply the TPS transformation
    try:
        x_new = rbf_x(x_flat, y_flat).reshape(rows, cols)
        y_new = rbf_y(x_flat, y_flat).reshape(rows, cols)
    except Exception as e:
        print(f"Error applying TPS transformation: {e}")
        return None, None

    # Remap source image to target using the new coordinates
    try:
        remapped_image = cv2.remap(source_image, x_new.astype(np.float32), y_new.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Error remapping image: {e}")
        return None, None

    return source_image, remapped_image

def plot_images_with_coordinates(source_image, target_image, transformed_image, source_coords, target_coords):
    plt.figure(figsize=(15, 5))

    # Plot source image with coordinates
    plt.subplot(1, 3, 1)
    plt.title('Source Image with Coordinates')
    plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    plt.scatter(*zip(*source_coords), color='red', s=1)  # Plot source coordinates
    plt.axis('off')

    # Plot target image with coordinates
    plt.subplot(1, 3, 2)
    plt.title('Target Image with Coordinates')
    plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    plt.scatter(*zip(*target_coords), color='red', s=1)  # Plot target coordinates
    plt.axis('off')

    # Plot transformed image
    plt.subplot(1, 3, 3)
    plt.title('Transformed Image')
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

# Example usage
source_image_path = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/source.jpg'
target_image_path = 'D:/WWU/M8 - Master Thesis/Project/Code/Images/target.jpg'

# Extract edge points
source_edge_coords = get_edge_points(source_image_path)
target_edge_coords = get_edge_points(target_image_path)

# Ensure we have corresponding points
if len(source_edge_coords) == 0 or len(target_edge_coords) == 0:
    print("Error: Edge points could not be extracted from one or both images.")
else:
    # Apply TPS transformation
    source_img, transformed_img = apply_tps_transformation(source_edge_coords, target_edge_coords, source_image_path)

    if source_img is not None and transformed_img is not None:
        # Load target image for plotting
        target_img = cv2.imread(target_image_path)

        # Plot the results with coordinates
        plot_images_with_coordinates(source_img, target_img, transformed_img, source_edge_coords, target_edge_coords)
    else:
        print("Error: TPS transformation failed.")
