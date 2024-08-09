import numpy as np
import random
import time
import sys
import pandas as pd
import keyboard
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from auto_connect import interactive_belt_connect, setup_logger
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption, BeltVibrationPattern)
from pybelt.belt_scanner import BeltScanner

from bracelet import connect_belt

connection_check, belt_controller = connect_belt()
if connection_check:
    print('Bracelet connection successful.')
else:
    print('Error connecting bracelet. Aborting.')
    sys.exit()

def interactive_belt_connect(belt_controller):
    """Interactive procedure to connect a belt. The interface to use is asked via the console.

    :param BeltController belt_controller: The belt controller to connect.
    """

    interface = 'u'
    if interface.lower() == "b":
        # Scan for advertising belt
        with pybelt.belt_scanner.create() as scanner:
            print("Start BLE scan.")
            belts = scanner.scan()
            print("BLE scan completed.")
        if len(belts) == 0:
            print("No belt found.")
            return belt_controller
        if len(belts) > 1:
            print("Select the belt to connect.")
            for i, belt in enumerate(belts):
                print("{}. {} - {}".format((i + 1), belt.name, belt.address))
            belt_selection = input("[1-{}]".format(len(belts)))
            try:
                belt_selection_int = int(belt_selection)
            except ValueError:
                print("Unrecognized input.")
                return belt_controller
            print("Connect the belt.")
            belt_controller.connect(belts[belt_selection_int - 1])
        else:
            print("Connect the belt.")
            belt_controller.connect(belts[0])

    elif interface.lower() == "u":
        # List serial COM ports
        ports = serial.tools.list_ports.comports()
        if ports is None or len(ports) == 0:
            print("No serial port found.")
            return belt_controller
        if len(ports) == 1:
            connect_ack = 'y'
            if connect_ack.lower() == "y" or connect_ack.lower() == "yes":
                print("Connect the belt.")
                belt_controller.connect(ports[0][0])
            else:
                print("Unrecognized input.")
                return belt_controller
        else:
            print("Select the serial COM port to use.")
            for i, port in enumerate(ports):
                print("{}. {}".format((i + 1), port[0]))
            belt_selection = input("[1-{}]".format(len(ports)))
            try:
                belt_selection_int = int(belt_selection)
            except ValueError:
                print("Unrecognized input.")
                return belt_controller
            print("Connect the belt.")
            belt_controller.connect(ports[belt_selection_int - 1][0])

    else:
        print("Unrecognized input.")
        return belt_controller

    return belt_controller

# Calibration function to determine optimal vibration intensity
def calibrate_intensity():
    intensity = 5
    while True:
        if belt_controller:
            belt_controller.send_vibration_command(
                channel_index=0,
                pattern=BeltVibrationPattern.CONTINUOUS,
                intensity=intensity,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=60,  # down
                pattern_iterations=None,
                pattern_period=500,
                pattern_start_time=0,
                exclusive_channel=False,
                clear_other_channels=False
            )
        print(f'Vibrating at intensity {intensity}.')
        user_input = input('Is this intensity sufficient? (yes/no): ').strip().lower()
        if user_input == 'yes':
            belt_controller.stop_vibration()     
            return intensity
        intensity += 5
        if intensity > 100:  # Maximum intensity cap
            print('Reached maximum intensity.')
            belt_controller.stop_vibration()     
            return intensity

# Calibrate the bracelet intensity
calibrated_intensity = calibrate_intensity()
print(f'Calibrated intensity: {calibrated_intensity}')

# Directions for training
directions = ['top', 'down', 'right', 'left', 'top right', 'bottom right', 'top left', 'bottom left']

# Function to send vibration for a given direction
def vibrate_direction(direction):
    if direction == 'top':
        belt_controller.send_vibration_command(            
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=90,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    elif direction == 'down':
        belt_controller.send_vibration_command(            
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=60,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    elif direction == 'right':
        belt_controller.send_vibration_command(           
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=120,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    elif direction == 'left':
        belt_controller.send_vibration_command(           
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.ANGLE,
            orientation=45,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    elif direction == 'top right':
        belt_controller.send_vibration_command(            
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b110000,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    elif direction == 'bottom right':
        belt_controller.send_vibration_command(
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b101000,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    elif direction == 'top left':
        belt_controller.send_vibration_command(           
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b010100,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)
    elif direction == 'bottom left':
        belt_controller.send_vibration_command(            
            channel_index=0,
            pattern=BeltVibrationPattern.CONTINUOUS,
            intensity=calibrated_intensity,
            orientation_type=BeltOrientationType.BINARY_MASK,
            orientation=0b001100,
            pattern_iterations=None,
            pattern_period=500,
            pattern_start_time=0,
            exclusive_channel=False,
            clear_other_channels=False)

# Function to capture the keyboard input for direction
def capture_direction():
    while True:
        if keyboard.is_pressed('up'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('right'):
                return 'top right'
            elif keyboard.is_pressed('left'):
                return 'top left'
            return 'top'
        elif keyboard.is_pressed('down'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('right'):
                return 'bottom right'
            elif keyboard.is_pressed('left'):
                return 'bottom left'
            return 'down'
        elif keyboard.is_pressed('right'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('up'):
                return 'top right'
            elif keyboard.is_pressed('down'):
                return 'bottom right'
            return 'right'
        elif keyboard.is_pressed('left'):
            time.sleep(0.1)  # Debounce delay
            if keyboard.is_pressed('up'):
                return 'top left'
            elif keyboard.is_pressed('down'):
                return 'bottom left'
            return 'left'

# Familiarization phase
def familiarization_phase():
    time.sleep(5)
    print("\nFamiliarization Phase")
    
    for direction in directions:
        while True:
            print(f"Vibrating for {direction}.")
            vibrate_direction(direction)
            user_response = capture_direction()
            print(f"User response: {user_response}")
            belt_controller.stop_vibration()
            time.sleep(1)  # Short delay between each trial
            
            if user_response == direction:
                break  # Exit the loop if the user response is correct
            else:
                print("Incorrect response. Please try again.")

# Training function
def training_task():
    print("\nTraining start will start")
    correct_responses_per_block = []
    blocks = 3
    trials_per_block = 16
    block_accuracies = []
    actual_directions =[]
    predicted_directions = []
    response_times = []

    for block in range(blocks):
        correct_responses = 0
        time.sleep(5)

        # Create a list with two of each direction and shuffle it
        block_directions = directions * 2
        random.shuffle(block_directions)

        for direction in range(block_directions):
            print(f"Trial {block * trials_per_block + trial + 1}: Vibration direction is {direction}.")
            vibrate_direction(direction)
            start_time = time.time()
            user_response = capture_direction()
            end_time = time.time()
            response_time = end_time - start_time

            print(f"User response: {user_response}")
            belt_controller.stop_vibration()
            time.sleep(1)
            actual_directions.append(direction)
            predicted_directions.append(user_response)
            response_times.append(response_time)

            if user_response == direction:
                correct_responses += 1

        # Stop vibration after completing a block with custom stop signal
        if belt_controller:
            belt_controller.stop_vibration()
            belt_controller.send_pulse_command(
                channel_index=0,
                orientation_type=BeltOrientationType.BINARY_MASK,
                orientation=0b111100,
                intensity=calibrated_intensity,
                on_duration_ms=150,
                pulse_period=500,
                pulse_iterations=5, 
                series_period=5000,
                series_iterations=1,
                timer_option=BeltVibrationTimerOption.RESET_TIMER,
                exclusive_channel=False,
                clear_other_channels=False)
            
        # Calculate accuracy for the block
        block_accuracy = (correct_responses / trials_per_block) * 100
        block_accuracies.append(block_accuracy)
        correct_responses_per_block.append(correct_responses)
        print(f"Block {block + 1} complete. Accuracy: {block_accuracy:.2f}%\n")
        
    # Calculate and display the average accuracy across all blocks
    average_accuracy = np.mean(block_accuracies)
    print(f"Selected intensity after training: {calibrated_intensity}")
    print(f"Block accuracy: {block_accuracies}")
    print(f"Training completed with an average accuracy of {average_accuracy:.2f}%")

    # Save result to Excel file
    results_df = pd.DataFrame({
        'Actual Direction': actual_directions,
        'Predicted Direction': predicted_directions,
        'Response Time (s)': response_times
    })
    results_df.to_excel('training_results.xlsx', index = False)
    print('\nResults saved to training_results.xlsx')

    # Determine if the training accuracy is sufficient
    if average_accuracy >= 90:
        print(f"Training completed with an accuracy of {average_accuracy:.2f}%")
    else: 
        print(f"Training accuracy below 90% with an accuracy of {average_accuracy:.2f}%")
        sys.exit()

    return average_accuracy, block_accuracies, actual_directions, predicted_directions


# Run familiarization phase
familiarization_phase()

# Run training task
training_accuracy = training_task()

