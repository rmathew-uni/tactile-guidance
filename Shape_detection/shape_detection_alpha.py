import numpy as np

# Define shapes with vertices
shapes = {
    'arrow': [(0, 0), (5, 0), (5, 1), (8, -1), (5, -3), (5, -2), (0, -2)],
    'cross': [(0, 0), (2, 0), (2, 2), (4, 2), (4, 0), (6, 0), (6, -2), (4, -2), (4, -4), (2, -4), (2, -2), (0, -2)],
    'hexagon': [(0, 0), (3, 2), (6, 0), (6, -3), (3, -5), (0, -3)],
    'kite': [(0, 0), (3, 2), (6, 0), (3, -6)],
    'octagon': [(0, 0), (2, 2), (4, 2), (6, 0), (6, -2), (4, -4), (2, -4)],
    'parallelogram': [(0, 0), (2, 2), (7, 2), (5, 0)],
    'pentagon': [(0, 0), (3, 2), (6, 0), (5, -3), (1, -3)],
    'rhombus': [(0, 0), (2, 2), (4, 2), (2, 0)],
    'star': [(0, 0), (2, 0), (3, 2), (4, 0), (6, 0), (4, -1), (5, -3), (3, -2), (1, -3), (2, -1)],
    'trapezoid': [(0, 0), (2, 2), (6, 2), (8, 0)],
    'square': [(0, 0), (0, 2), (2, 2), (2, 0)],
    'rectangle': [(0, 0), (0, 2), (4, 2), (4, 0)],
    'triangle': [(0, 0), (3, 3), (3, 0)],
    'one' : [(0,0), (0,-8)],
    'two' : [(0,0), (4,0), (4,-4), (0,-4), (0,-8), (4,-8)],
    'three' : [(0,0), (4,0), (4,-4), (0,-4), (4,-4), (4,-8), (0,-8)],
    'four' : [(0,0), (0,-4), (4,-4), (4,0), (4,-8)],
    'five' : [(0,0), (-4,0), (-4,-4), (0,-4), (0,-8), (-4,-8)],
    'six' : [(0,0), (-4,0), (-4,-8), (0,-8), (0,-4), (-4,-4)],
    'seven' : [(0,0), (4,0), (4,-8)],
    'eight' : [(0,0), (-4,0), (-4,-4), (0,-4), (0,-8), (-4,-8), (-4,-4), (0,-4), (0,0)],
    'nine' : [(0,0), (-4,0), (-4,-4), (0,-4), (0,0), (0,-8), (-4,-8)],
    'zero' : [(0,0), (4,0), (4,-8), (0,-8), (0,0)],
}

# Function to calculate direction and distance
def calculate_direction_and_distance(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    if dx > 0 and dy == 0:
        return 'right', distance
    elif dx < 0 and dy == 0:
        return 'left', distance
    elif dy > 0 and dx == 0:
        return 'top', distance
    elif dy < 0 and dx == 0:
        return 'down', distance
    elif dx > 0 and dy > 0:
        return 'diagonal right top', distance
    elif dx > 0 and dy < 0:
        return 'diagonal right bottom', distance
    elif dx < 0 and dy > 0:
        return 'diagonal left top', distance
    elif dx < 0 and dy < 0:
        return 'diagonal left bottom', distance
    else:
        return 'none', 0

# Function to detect outline and simulate tactile feedback
def simulate_tactile_feedback(shape):
    vertices = shapes[shape]
    vertices.append(vertices[0])  # Close the shape
    
    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        direction, distance = calculate_direction_and_distance(start, end)
        if direction != 'none':
            duration = distance  # Assuming duration is proportional to distance
            print(f"Move {direction} for {duration:.2f} units")

# Loop to input multiple shapes
while True:
    shape_to_detect = input("Enter the shape to detect (or 'exit' to stop): ").strip().lower()
    if shape_to_detect == 'exit':
        break
    simulate_tactile_feedback(shape_to_detect)
    print("\n")  # Adding a newline for better readability between inputs

'''



import numpy as np

# Define shapes with vertices
shapes = {
    'arrow': [(0, 0), (5, 0), (5, 1), (8, -1), (5, -3), (5, -2), (0, -2)],
    'cross': [(0, 0), (2, 0), (2, 2), (4, 2), (4, 0), (6, 0), (6, -2), (4, -2), (4, -4), (2, -4), (2, -2), (0, -2)],
    'hexagon': [(0, 0), (3, 2), (6, 0), (6, -3), (3, -5), (0, -3)],
    'kite': [(0, 0), (3, 2), (6, 0), (3, -6)],
    'octagon': [(0, 0), (2, 2), (4, 2), (6, 0), (6, -2), (4, -4), (2, -4)],
    'parallelogram': [(0, 0), (2, 2), (7, 2), (5, 0)],
    'pentagon': [(0, 0), (3, 2), (6, 0), (5, -3), (1, -3)],
    'rhombus': [(0, 0), (2, 2), (4, 2), (2, 0)],
    'star': [(0, 0), (2, 0), (3, 2), (4, 0), (6, 0), (4, -1), (5, -3), (3, -2), (1, -3), (2, -1)],
    'trapezoid': [(0, 0), (2, 2), (6, 2), (8, 0)],
    'square': [(0, 0), (0, 1), (1, 1), (1, 0)],
    'square': [(0, 0), (0, 2), (2, 2), (2, 0)],
    'rectangle': [(0, 0), (0, 2), (4, 2), (4, 0)],
    'triangle': [(0, 0), (3, 3), (3, 0)],
}

# Function to calculate direction and distance
def calculate_direction_and_distance(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    if abs(dx) >= abs(dy):  # Horizontal movement
        if dx > 0:
        if belt_controller:
            belt_controller.vibrate_at_angle(120, channel_index=0, intensity=vibration_intensity)
            return 'right', distance
        elif dx < 0:
        if belt_controller:
            belt_controller.vibrate_at_angle(45, channel_index=0, intensity=vibration_intensity)
            return 'left', distance
    else:  # Vertical movement
        if dy > 0:
         if belt_controller:
            belt_controller.vibrate_at_angle(90, channel_index=0, intensity=vibration_intensity)
            return 'top', distance
        elif dy < 0:
        if belt_controller:
            belt_controller.vibrate_at_angle(60, channel_index=0, intensity=vibration_intensity)
            return 'down', distance

    return 'none', 0

# Function to detect outline and simulate tactile feedback
def simulate_tactile_feedback(shape):
    vertices = shapes.get(shape)
    if vertices is None:
        print(f"Shape '{shape}' not found.")
        return
    
    vertices.append(vertices[0])  # Close the shape
    
    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        direction, distance = calculate_direction_and_distance(start, end)
        if direction != 'none':
            duration = distance  # Assuming duration is proportional to distance
            print(f"Move {direction} for {duration:.2f} units")

# Loop to input multiple shapes
while True:
    shape_to_detect = input("Enter the shape to detect (or 'exit' to stop): ").strip().lower()
    if shape_to_detect == 'exit':
        break
    simulate_tactile_feedback(shape_to_detect)
    print("\n")  # Adding a newline for better readability between inputs
