import os
import cv2
import numpy as np

def get_direction(start_point, end_point):
    """
    Determine the direction from start_point to end_point.
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    if dx > 0 and dy == 0:
        return 'right'
    elif dx < 0 and dy == 0:
        return 'left'
    elif dx == 0 and dy > 0:
        return 'down'
    elif dx == 0 and dy < 0:
        return 'up'
    elif dx > 0 and dy < 0:
        return 'diagonal right top'
    elif dx < 0 and dy < 0:
        return 'diagonal left top'
    elif dx > 0 and dy > 0:
        return 'diagonal right bottom'
    elif dx < 0 and dy > 0:
        return 'diagonal left bottom'

def detect_shapes_and_generate_tactile_commands(image_path):
    """
    Function to detect shapes and generate tactile feedback commands for the entire contour.

    Parameters:
    - image_path (str): Path to the input image file.

    Returns:
    - list: List of tuples (commands) for each detected shape.
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return []

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to obtain binary image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tactile_commands = []

    # Process each contour
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Generate direction commands for the contour
        commands = []
        for i in range(len(approx)):
            start_point = approx[i][0]
            end_point = approx[(i + 1) % len(approx)][0]
            command = get_direction(start_point, end_point)
            commands.append(command)

            # Add a 'stop' command when the last line meets the first line
            if (i + 1) % len(approx) == 0:
                commands.append('stop')

        tactile_commands.append(commands)

        # Print tactile commands
        print(f"Tactile Feedback Commands: {commands}")

    return tactile_commands

def process_specified_images(folder_path, image_files):
    """
    Function to process specified images from a folder.

    Parameters:
    - folder_path (str): Path to the folder containing images.
    - image_files (list): List of image file names to process.
    """
    # List all files in the folder to ensure they are correct
    print("Listing all files in the folder:")
    all_files = os.listdir(folder_path)
    print(all_files)

    # Process each specified image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        if not os.path.exists(image_path):
            print(f"File does not exist: {image_path}")
            continue

        print(f"\nProcessing Image: {image_path}")
        shape_commands = detect_shapes_and_generate_tactile_commands(image_path)
        
        # Print tactile feedback commands for each detected shape
        for commands in shape_commands:
            print(f"Tactile Feedback Commands: {commands}")

# Example usage:
folder_path = 'C:/Users/Felicia/Bracelet/tactile-guidance-main/Shape detection/Images/'
image_files = ['pentagon.jpg']  # Specify the image file(s) to process

process_specified_images(folder_path, image_files)
