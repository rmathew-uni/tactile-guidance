import numpy as np
import time
import sys
from auto_connect import interactive_belt_connect, setup_logger
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption, BeltVibrationPattern)

from bracelet import connect_belt

connection_check, belt_controller = connect_belt()
if connection_check:
    print('Bracelet connection successful.')
else:
    print('Error connecting bracelet. Aborting.')
    sys.exit()
                                    
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

# Function to calculate direction and distance
def calculate_direction_and_time(start, end, speed=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = np.sqrt(dx**2 + dy**2)
    time_required = distance / speed 

    vibration_intensity = 50
    
    if dx > 0 and dy == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=120,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'right', time_required
    elif dx < 0 and dy == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=45,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'left', time_required
    elif dy > 0 and dx == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=90,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )     
            return 'top', time_required
    elif dy < 0 and dx == 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=60,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'down', time_required
    elif dx > 0 and dy > 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b110000,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal right top', time_required
    elif dx > 0 and dy < 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b101000,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal right bottom', time_required
    elif dx < 0 and dy > 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b010100,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal left top', time_required
    elif dx < 0 and dy < 0:
        if belt_controller:
            belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=vibration_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b001100,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False
            )
            return 'diagonal left bottom', time_required
    else:
        return 'none', 0
    
# Function to detect outline and simulate tactile feedback
def simulate_tactile_feedback(shape, speed=1):
    vertices = shapes[shape]
    if shape in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 
                 'c', 'e', 'j', 'l', 'm', 'n', 'p', 'u', 'r', 'v', 'w', 'z']:
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
shapes_to_detect_1 = ['square', 'octagon', 'cross', 'seven', 'diamond', 'one', 'triangle', 
                      'two', 'w', 'kite', 'rectangle', 'nine', 'c', 'j', 'four', 'star', 'n', 
                      'pentagon', 'p', 'z', 'hexagon', 'u', 'l', 'three', 'v', 'six', 'e', 
                      'hourglass', 'm', 'eight', 'five', 'parallelogram', 'r', 'arrow', 'trapezoid']

shapes_to_detect_2 = ['parallelogram', 'e', 'arrow', 'r', 'six', 'p', 'one', 'seven', 'square', 
                      'u', 'n', 'pentagon', 'diamond', 'j', 'three', 'v', 'triangle', 'star', 'm', 
                      'five', 'rectangle', 'four', 'hexagon', 'kite', 'nine', 'octagon', 'eight', 
                      'w', 'trapezoid', 'cross', 'z', 'hourglass', 'c', 'l', 'two']

shapes_to_detect_3 = ['two', 'hexagon', 'n', 'l', 'cross', 'arrow', 'r', 'nine', 'eight', 'm', 
                      'seven', 'kite', 'rectangle', 'c', 'three', 'u', 'hourglass', 'five', 'star', 
                      'six', 'e', 'diamond', 'square', 'j', 'parallelogram', 'trapezoid', 'pentagon', 
                      'w', 'octagon', 'p', 'z', 'four', 'one', 'v', 'triangle']

# Loop through the shapes
for index, shape in enumerate(shapes_to_detect_1):
    time.sleep(3)
    print(shape)
    simulate_tactile_feedback(shape)
    print("stop \n")  # Adding a newline for better readability between shapes
    if belt_controller:
        belt_controller.stop_vibration()
        belt_controller.send_pulse_command(
            channel_index=0,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b111100,
            intensity=100,
            on_duration_ms=150,
            pulse_period=500,
            pulse_iterations=5, 
            series_period=5000,
            series_iterations=1,
            timer_option=BeltVibrationTimerOption.RESET_TIMER,
            exclusive_channel=False,
            clear_other_channels=False)
    time.sleep(4)  # Pause for 3 seconds after each shape
    
    # Add a 5-second rest after every 5 shapes
    if (index + 1) % 5 == 0:
        print("5-second rest \n")
        time.sleep(5)


# Example usage
# shape_to_detect = 'nine'  # Change to 'rectangle', 'triangle', 'polygon' as needed
# simulate_tactile_feedback(shape_to_detect)
