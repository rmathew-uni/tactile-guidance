import numpy as np
import time
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption)

# Define shapes with vertices
shapes = {
    'arrow': [(0, 0), (4, 0), (4, 1), (7, -1), (4, -3), (4, -2), (0, -2)],
    'cross': [(0, 0), (2, 0), (2, 2), (4, 2), (4, 0), (6, 0), (6, -2), (4, -2), (4, -4), (2, -4), (2, -2), (0, -2)],
    'hexagon': [(0, 0), (3, 2), (6, 0), (6, -3), (3, -5), (0, -3)],
    'kite': [(0, 0), (3, 2), (6, 0), (3, -6)],
    'octagon': [(0, 0), (2, 2), (4, 2), (6, 0), (6, -2), (4, -4), (2, -4)],
    'parallelogram': [(0, 0), (2, 2), (6, 2), (4, 0)],
    'pentagon': [(0, 0), (3, 2), (6, 0), (5, -3), (1, -3)],
    'rhombus': [(0, 0), (2, 2), (4, 2), (2, 0)],
    'star': [(0, 0), (2, 0), (3, 2), (4, 0), (6, 0), (4, -1), (5, -3), (3, -2), (1, -3), (2, -1)],
    'trapezoid': [(0, 0), (2, 2), (6, 2), (8, 0)],
    'square': [(0, 0), (0, 2), (2, 2), (2, 0)],
    'rectangle': [(0, 0), (0, 2), (4, 2), (4, 0)],
    'triangle': [(0, 0), (3, 3), (3, 0)],
    'diamond' :[(0,0), (2,2), (6,2), (8,0), (4,-6)],
    'one' : [(0,0), (0,-4)],
    'two' : [(0,0), (2,0), (2,-2), (0,-2), (0,-4), (2,-4)],
    'three' : [(0,0), (2,0), (2,-2), (0,-2), (2,-2), (2,-4), (0,-4)],
    'four' : [(0,0), (0,-2), (2,-2), (2,0), (2,-4)],
    'five' : [(0,0), (-2,0), (-2,-2), (0,-2), (0,-4), (-2,-4)],
    'six' : [(0,0), (-2,0), (-2,-4), (0,-4), (0,-2), (-2,-2)],
    'seven' : [(0,0), (2,0), (2,-4)],
    'eight' : [(0,0), (-2,0), (-2,-2), (0,-2), (0,-4), (-2,-4), (-2,-2), (0,-2), (0,0)],
    'nine' : [(0,0), (-2,0), (-2,-2), (0,-2), (0,0), (0,-4), (-2,-4)],
}

# Function to calculate direction and distance
def calculate_direction_and_time(start, end, speed=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    time_required = distance / speed 
    
    if dx > 0 and dy == 0:
        if belt_controller:
            belt_controller.vibrate_at_angle(120, channel_index=0, intensity=vibration_intensity)
            return 'right', time_required
    elif dx < 0 and dy == 0:
        if belt_controller:
            belt_controller.vibrate_at_angle(45, channel_index=0, intensity=vibration_intensity)
            return 'left', time_required
    elif dy > 0 and dx == 0:
        if belt_controller:
            belt_controller.vibrate_at_angle(90, channel_index=0, intensity=vibration_intensity)        
            return 'top', time_required
    elif dy < 0 and dx == 0:
        if belt_controller:
            belt_controller.vibrate_at_angle(60, channel_index=0, intensity=vibration_intensity)
            return 'down', time_required
    elif dx > 0 and dy > 0:
        if belt_controller:
            belt_controller.send_pulse_command(
                channel_index=0,
                orientation_type=BeltOrientationType.BINARY_MASK,
                orientation=0b111100)
            return 'diagonal right top', time_required
    elif dx > 0 and dy < 0:
        if belt_controller:
            belt_controller.send_pulse_command(
                channel_index=0,
                orientation_type=BeltOrientationType.BINARY_MASK,
                orientation=0b111100)
            return 'diagonal right bottom', time_required
    elif dx < 0 and dy > 0:
        if belt_controller:
            belt_controller.send_pulse_command(
                channel_index=0,
                orientation_type=BeltOrientationType.BINARY_MASK,
                orientation=0b111100)
            return 'diagonal left top', time_required
    elif dx < 0 and dy < 0:
        if belt_controller:
            belt_controller.send_pulse_command(
                channel_index=0,
                orientation_type=BeltOrientationType.BINARY_MASK,
                orientation=0b111100)
            return 'diagonal left bottom', time_required
    else:
        return 'none', 0
    
# Function to detect outline and simulate tactile feedback
def simulate_tactile_feedback(shape, speed=1):
    vertices = shapes[shape]
    if shape in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
        vertices.append(vertices[-1])  # Add the last vertex again to complete the shape
    else:
        vertices.append(vertices[0])  # Close the shape by returning to the starting point
    
    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        direction, time_required = calculate_direction_and_time(start, end, speed)
        if direction != 'none':
            print(f"Move {direction} for {time_required:.2f} seconds")
            time.sleep(time_required)  # Simulate the time required for the movement

# List of shapes to loop through
shapes_to_detect = ['parallelogram', 'square', 'nine',	'hexagon',	'pentagon',	
                    'arrow',	'cross',	'rhombus',	'triangle','one',	'star',	
                    'three',	'two',	'rectangle',	'five',	'seven',	'kite',	
                    'four',	'octagon',	'trapezoid',	'six',	'diamond', 'eight']

'''
shapes_to_detect = ['rhombus',	'seven',	'kite',	'two',	'octagon',	'star',	
                    'triangle', 'one',	, 'diamond', 'eight',	'three',	'six',	'arrow',	
                    'four',	'hexagon',	'pentagon',	'five',	'nine',	'square',	'cross',
                    'parallelogram',	'trapezoid',	'rectangle']

shapes_to_detect = ['five',	'hexagon', 'kite', 'octagon', 'square',	'cross',	
                    'nine',	'parallelogram',	'trapezoid', 'six',	'pentagon',	
                    'two',	'seven',	'rectangle',	'triangle', 'one',	, 'diamond', 
                    'four',	'star',	'eight',	'arrow',	'rhombus',	'three']

'''

# Loop through the shapes
for index, shape in enumerate(shapes_to_detect):
    print(shape)
    simulate_tactile_feedback(shape)
    print("stop \n")  # Adding a newline for better readability between shapes
    if belt_controller:
        belt_controller.stop_vibration()
        belt_controller.send_pulse_command(
            channel_index=0,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b111100,
            pulse_iterations=5)
    time.sleep(2)  # Pause for 2 seconds after each shape
    
    # Add a 5-second rest after every 5 shapes
    if (index + 1) % 5 == 0:
        print("5-second rest \n")
        time.sleep(5)


# Example usage
# shape_to_detect = 'nine'  # Change to 'rectangle', 'triangle', 'polygon' as needed
# simulate_tactile_feedback(shape_to_detect)
