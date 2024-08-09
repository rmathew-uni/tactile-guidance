import numpy as np
import time
import sys
import random

# Define shapes with vertices
shapes = {
    '0': [(0, 0), (0, 4), (2, 4), (2, 0), (0, 0)],
    '1': [(0, 0), (0, -2)],
    '2': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4)],
    '3': [(0, 0), (2, 0), (2, -2), (0, -2), (2, -2), (2, -4), (0, -4)],
    '4': [(0, 0), (0, -2), (2, -2), (2, 0), (2, -4)],
    '5': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4)],
    '6': [(0, 0), (-2, 0), (-2, -4), (0, -4), (0, -2), (-2, -2)],
    '7': [(0, 0), (2, 0), (2, -4)],
    '8': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4), (-2, -2), (0, -2), (0, 0)],
    '9': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -4), (-2, -4)],
    'a': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -2.5)],
    'b': [(0, 0), (0, -4), (2, -4), (2, -2), (0, -2)],
    'c': [(0, 0), (-2, 0), (-2, -2), (0, -2)],
    'd': [(0, 0), (0, -4), (-2, -4), (-2, -2), (0, -2)],
    'e': [(0, 0), (2, 0), (2, 2), (0, 2), (0, -2), (2, -2)],
    'f': [(0, 0), (-2, 0), (-2, -4), (-2, -2), (0, -2)],
    'h': [(0, 0), (0, -4), (0, -2), (2, -2), (2, -4)],
    'i': [(0, 0), (4, 0), (2, 0), (2, -4), (0, -4), (4, -4)],
    'j': [(0, 0), (2, 0), (2, -4), (0, -4), (0, -2)],
    'k': [(0, 0), (0, -4), (2, -2), (1, -3), (2, -4)],
    'l': [(0, 0), (0, -4), (2, -4)],
    'm': [(0, 0), (0, 4), (2, 2), (4, 4), (4, 0)],
    'n': [(0, 0), (0, 4), (2, 0), (2, 4)],
    'p': [(0, 0), (0, 4), (2, 4), (2, 2), (0, 2)],
    'q': [(0, 0), (0, 4), (-2, 4), (-2, 2), (0, 2)],
    'u': [(0, 0), (0, -2), (2, -2), (2, 0)],
    'r': [(0, 0), (0, 4), (2, 4), (2, 2), (0, 2), (2, 0)],
    'v': [(0, 0), (2, -4), (4, 0)],
    'w': [(0, 0), (0, -4), (2, -2), (4, -4), (4, 0)],
    'y': [(0, 0), (2, -2), (4, 0), (0, -4)],
    'z': [(0, 0), (2, 0), (0, -2), (2, -2)]
}

# Function to calculate direction and distance
def calculate_direction_and_time(start, end, speed=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    time_required = distance / speed 

    vibration_intensity = 50
    
    if dx > 0 and dy == 0:
        return 'right', time_required
    elif dx < 0 and dy == 0:
        return 'left', time_required
    elif dy > 0 and dx == 0: 
         return 'top', time_required
    elif dy < 0 and dx == 0:
        return 'down', time_required
    elif dx > 0 and dy > 0:
        return 'diagonal right top', time_required
    elif dx > 0 and dy < 0:
        return 'diagonal right bottom', time_required
    elif dx < 0 and dy > 0:
        return 'diagonal left top', time_required
    elif dx < 0 and dy < 0:
        return 'diagonal left bottom', time_required
    else:
        return 'none', 0
    

# Function to simulate tactile feedback based on shape
def simulate_tactile_feedback(shape, speed=1):
    vertices = shapes[shape]
    vertices.append(vertices[-1])  # Add the last vertex again to complete the shape

    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        direction, time_required = calculate_direction_and_time(start, end, speed)
        if direction != 'none':
            print(f"{direction} for {time_required:.2f} seconds")
            time.sleep(time_required)  # Simulate the time required for the movement

# Define the categories and their items
categories = {
    'numbers': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'letters': ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'l', 'p', 'q', 'u'],
    'beta': ['k', 'm', 'n', 'r', 'v', 'w', 'y', 'z']
}

# Shuffle the items within each category for each participant
for category, items in categories.items():
    random.shuffle(items)

# Execute drawing tasks for each category sequentially
for category, items in categories.items():
    print(f"Starting category: {category}")
    for index, item in enumerate(items):
        time.sleep(3)
        print(item)
        simulate_tactile_feedback(item)
        print("stop \n")
        time.sleep(4)  # Pause after each shape

        # Add a 5-second rest after every 5 items within the category
        if (index + 1) % 5 == 0:
            print("5-second rest \n")
            time.sleep(5)