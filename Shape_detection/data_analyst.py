import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_contour_coordinates(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply edge detection
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract coordinates
    coordinates = []
    for contour in contours:
        for point in contour:
            coordinates.append(tuple(point[0]))
    
    return coordinates

def scale_coordinates(template_coords, target_coords):
    template_coords = np.array(template_coords)
    target_coords = np.array(target_coords)
    
    template_min, template_max = np.min(template_coords, axis=0), np.max(template_coords, axis=0)
    target_min, target_max = np.min(target_coords, axis=0), np.max(target_coords, axis=0)
    
    scale_factor = (target_max - target_min) / (template_max - template_min)
    scaled_template_coords = (template_coords - template_min) * scale_factor + target_min
    
    return scaled_template_coords

def align_coordinates(coords1, coords2):
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    aligned_coords = coords1 - center1 + center2
    
    return aligned_coords

def calculate_error(scaled_coords, target_coords):
    error = np.linalg.norm(scaled_coords - target_coords, axis=1)
    mean_error = np.mean(error)
    
    return mean_error

# Example usage with the template and participant images
template_path = 'path_to_template_image.png'
participant_path = 'path_to_participant_image.png'

template_coords = get_contour_coordinates(template_path)
participant_coords = get_contour_coordinates(participant_path)

# Scale and align coordinates
scaled_template_coords = scale_coordinates(template_coords, participant_coords)
aligned_template_coords = align_coordinates(scaled_template_coords, participant_coords)

# Calculate error
error = calculate_error(aligned_template_coords, participant_coords)

# Plot to visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Template vs. Participant")
plt.scatter(*zip(*template_coords), label='Template', c='b')
plt.scatter(*zip(*participant_coords), label='Participant', c='r')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Scaled and Aligned Template vs. Participant")
plt.scatter(*zip(*aligned_template_coords), label='Scaled Template', c='b')
plt.scatter(*zip(*participant_coords), label='Participant', c='r')
plt.legend()

plt.show()

print(f'Mean Error: {error}')
