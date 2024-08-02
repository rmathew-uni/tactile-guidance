import matplotlib.pyplot as plt
import numpy as np

# Define shapes with vertices
shapes = {
    'arrow': [(0, 0), (3, 0), (3, 2), (6, -1), (3, -4), (3, -2), (0, -2)],
    'cross': [(0, 0), (2, 0), (2, 2), (4, 2), (4, 0), (6, 0), (6, -2), (4, -2), (4, -4), (2, -4), (2, -2), (0, -2)],
    'hexagon': [(0, 0), (3, 2), (6, 0), (6, -3), (3, -5), (0, -3)],
    'kite': [(0, 0), (2, 2), (4, 0), (2, -5)],
    'octagon': [(0, 0), (2, 2), (4, 2), (6, 0), (6, -2), (4, -4), (2, -4), (0, -2)],
    'parallelogram': [(0, 0), (2, 2), (6, 2), (4, 0)],
    'pentagon': [(0, 0), (2, 2), (4, 0), (3, -2), (1, -2)],
    'hourglass': [(0, 0), (4, 0), (3, 3), (4, 6), (0, 6), (1, 3)],
    'star': [(0, 0), (3, 5), (6, 0), (0, 3), (6, 3)],
    'trapezoid': [(0, 0), (1, 2), (4, 2), (5, 0)],
    'square': [(0, 0), (0, 2), (2, 2), (2, 0)],
    'rectangle': [(0, 0), (0, 2), (4, 2), (4, 0)],
    'triangle': [(0, 0), (3, 3), (3, 0)],
    'diamond': [(0, 0), (2, 2), (6, 2), (8, 0), (4, -4)],
    'one': [(0, 0), (0, -4)],
    'two': [(0, 0), (2, 0), (2, -2), (0, -2), (0, -4), (2, -4)],
    'three': [(0, 0), (2, 0), (2, -2), (0, -2), (2, -2), (2, -4), (0, -4)],
    'four': [(0, 0), (0, -2), (2, -2), (2, 0), (2, -4)],
    'five': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4)],
    'six': [(0, 0), (-2, 0), (-2, -4), (0, -4), (0, -2), (-2, -2)],
    'seven': [(0, 0), (2, 0), (2, -4)],
    'eight': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, -4), (-2, -4), (-2, -2), (0, -2), (0, 0)],
    'nine': [(0, 0), (-2, 0), (-2, -2), (0, -2), (0, 0), (0, -4), (-2, -4)],
    'c': [(0, 0), (-2, 0), (-2, -2), (0, -2)],
    'e': [(0, 0), (-2, 0), (-2, -2), (0, -2), (-2, -2), (-2, -4), (0, -4)],
    'j': [(0, 0), (2, 0), (2, -4), (0, -4), (0, -2)],
    'l': [(0, 0), (0, -4), (2, -4)],
    'm': [(0, 0), (0, 2), (2, 2), (2, 0), (2, 2), (4, 2), (4, 0)],
    'n': [(0, 0), (0, 2), (2, 2), (2, 0)],
    'p': [(0, 0), (0, 4), (2, 4), (2, 2), (0, 2)],
    'u': [(0, 0), (0, -2), (2, -2), (2, 0)],
    'r': [(0, 0), (0, 4), (2, 4), (2, 2), (0, 2), (2, 0)],
    'v': [(0, 0), (2, -4), (4, 0)],
    'w': [(0, 0), (0, -4), (2, -2), (4, -4), (4, 0)],
    'z': [(0, 0), (2, 0), (0, -2), (2, -2)]
}

# Function to calculate direction and time
def calculate_direction_and_time(start, end, speed=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    time_required = distance / speed
    return '', time_required  # Simplified for visualization

# Function to visualize shapes
def visualize_shapes(shapes, speed=1):
    for shape_name, vertices in shapes.items():
        plt.figure()
        plt.title(f"Shape: {shape_name}")
        
        if shape_name in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 
                          'c', 'e', 'j', 'l', 'm', 'n', 'p', 'u', 'r', 'v', 'w', 'z']:
            vertices.append(vertices[-1])  # Add the last vertex again to complete the shape
        else:
            vertices.append(vertices[0])  # Close the shape by returning to the starting point
        
        for i in range(len(vertices) - 1):
            start = vertices[i]
            end = vertices[i + 1]
            direction, time_required = calculate_direction_and_time(start, end, speed)
            
            # Plot the line segment
            plt.plot([start[0], end[0]], [start[1], end[1]], marker='o')
            plt.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, f'{time_required:.2f}s', fontsize=9)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

# Visualize all shapes
visualize_shapes(shapes)