import numpy as np
import random
import time
import sys
import keyboard
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
    print("Familiarization Phase: Please press the corresponding arrow keys for each vibration.")
    for direction in directions:
        print(f"Vibrating for {direction}.")
        vibrate_direction(direction)
        user_response = capture_direction()
        print(f"User response: {user_response}")
        belt_controller.stop_vibration()
        time.sleep(1)  # Short delay between each familiarization trial

# Training function
def training_task():
    correct_responses_per_block = []
    blocks = 3
    trials_per_block = 16

    for block in range(blocks):
        correct_responses = 0
        for trial in range(trials_per_block):
            direction = random.choice(directions)
            print(f"Trial {block * trials_per_block + trial + 1}: Vibration direction is {direction}.")
            vibrate_direction(direction)
            user_response = capture_direction()
            print(f"User response: {user_response}")
            if user_response == direction:
                correct_responses += 1
            time.sleep(1)  # Short delay between trials
        
        # Stop vibration after completing a block with custom stop signal
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

        # Calculate accuracy for the block
        block_accuracy = (correct_responses / trials_per_block) * 100
        correct_responses_per_block.append(block_accuracy)
        print(f"Block {block + 1} complete. Accuracy: {block_accuracy:.2f}%")

    # Calculate and display the average accuracy across all blocks
    average_accuracy = sum(correct_responses_per_block) / blocks
    print(f"Training completed with an average accuracy of {average_accuracy:.2f}%")
    return average_accuracy

# Run familiarization phase
familiarization_phase()

# Run training task
training_accuracy = training_task()
if training_accuracy >= 90:
    print(f"Training completed with an accuracy of {training_accuracy:.2f}%")
else:
    print(f"Training accuracy below 90% with an accuracy of {training_accuracy:.2f}%")
    sys.exit()
