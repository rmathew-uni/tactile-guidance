import keyboard
import uuid

import time
import pygame
from pathlib import Path
import csv

from connect import interactive_belt_connect, setup_logger

from pybelt.belt_controller import BeltController, BeltConnectionState, BeltControllerDelegate, BeltMode, \
    BeltOrientationType, BeltVibrationTimerOption


class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass

pygame.init()

# Global variables
participantID = 0
belt_controller_delegate = Delegate()
belt_controller = BeltController(belt_controller_delegate)
obs_loacalization = list(([],[],[]))
obs_grasping = [["time", "num_instructions", "condition"]]
#obs_graping_audio = list([])


def main():
   
    print("Welcome to the \"Guiding grasping motions of blindfolded subjects using localized tactile stimulation\" experiment!")
    print("Please connect the tactile bracelet and press Enter to continue.")

    # Wait for user to continue.
    while True: 
        if keyboard.is_pressed("enter"):
            break

    belt_controller = connect_belt()

    # Generate a new Participant ID or enter an existing one.
    user_in = ""
    while True:
       
        if user_in == "y":
            participantID = str(uuid.uuid4())
            print(participantID)
            break

        elif user_in == "n":
            #print()
            participantID = input("Please enter a participant ID: ")
            print(participantID)
            break
        
        print("Generate new Participant ID? [y,n]")
        user_in = input()

    
    print("Press 1 to start the localization task. Press 2 to start the grasping task.")

    #select localization or grasping task.
    while True:
        print("localization or graspin? [1,2]")
        user_in = input()
        print(user_in)

        # Start localization task
        if user_in == "1":
            localization_task()
            print(obs_loacalization)

            write_to_csv(participantID, obs_loacalization, "localization")
            
            # Save observations to csv file.

            #if file exists read rows into list so that new obs can be added to old
            #filepath = str(participantID + ".csv")
            #with open(filepath, 'w') as file:
                #writer = csv.writer(file)

                #for list in obs_loacalization:
                    #writer.writerow(list)

                

        # Start grasping task
        if user_in == "2":
            user_in = ""
            while user_in != "1" and user_in != "2":
                user_in = input("tactile or auditory? [1,2]")
                if user_in == "1":
                    grasping_task_tactile()
                    print(obs_grasping)
                    write_to_csv(participantID, obs_grasping, "grasping")
                elif user_in == "2":
                    grasping_task_auditory()
                    print(obs_grasping)
                    write_to_csv(participantID, obs_grasping, "grasping")
                
        

def write_to_csv(id, observations, task):

    #if file exists read rows into list so that new obs can be added to old
    id = id
    obs = observations
    if task == "localization":
        filepath = str(id + "_localization" + ".csv")

    elif task == "grasping":
        filepath = str(id + "_grasping" + ".csv")

    with open(filepath, 'w') as file:
        writer = csv.writer(file)
        for list in obs:
            writer.writerow(list)


def connect_belt():
    setup_logger()

    # Interactive script to connect the belt
    
    interactive_belt_connect(belt_controller)
    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Connection failed.")
        return 0

    # Change belt mode to APP mode
    belt_controller.set_belt_mode(BeltMode.APP_MODE)
    


def localization_task():
    while belt_controller.get_connection_state() == BeltConnectionState.CONNECTED:
        print("Q to quit.")
        print("0. Stop vibration.")
        print("1. Trials block 1")
        print("2. Trials block 2")
        print("3. Trials block 3")
        action = input()
        while True:
            try:
                action_int = int(action)
                if action_int == 0:
                    belt_controller.stop_vibration()
                    print("stop")
                    break
                elif action_int == 1:
                    #1. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )

                    #time bounded inupt
                    obs_loacalization[0].append(collect_response())
            

                    #wait for input
                    #time.sleep(3)
                    #2. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=2000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    #time.sleep(3)
                    #3. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    print(obs_loacalization)
                    #time.sleep(3)
                    #4. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 5. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 6. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 7. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 8. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 9. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 10. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 11. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 12. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 13. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 14. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 15. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    # 16. RIght
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[0].append(collect_response())
                    break

                elif action_int == 2:
                    # 1. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 2. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 3. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 4. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 4. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 6. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 7. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 8. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 9. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 10. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 11. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 12. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 13. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 14. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 15. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    # 16. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[1].append(collect_response())
                    break
                elif action_int == 3:
                    # 1. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 2. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=2000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 3. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 4. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 5. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 6. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 7. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 8. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 9. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 10. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 11. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 12. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 13. Up
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 14. Right
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=120,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 15. Left
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=45,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    # 16. Down
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=60,
                        intensity=None,
                        on_duration_ms=500,
                        pulse_period=1000,
                        pulse_iterations=1,
                        series_period=1500,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    obs_loacalization[2].append(collect_response())
                    break
                else:
                    print("Unrecognized input.")
                    break
            except ValueError:
                if action.lower() == "q" or action.lower() == "quit":
                    #belt_controller.disconnect_belt()
                    return
                else:
                    print("Unrecognized input.")
                    break

def grasping_task_tactile():
    new_trial = True
    num_instructions = 0
    curr = ""
    last = ""
    while belt_controller.get_connection_state() == BeltConnectionState.CONNECTED:

        while True:
            try:
                print(num_instructions)
                # Check if its the first instruction and if it is log the start time.
                if (keyboard.is_pressed("left") or keyboard.is_pressed("right") or keyboard.is_pressed("up") or keyboard.is_pressed("down")) and new_trial:
                    begin = time.time()
                    print(begin)
                    #num_instructions += 1
                    new_trial = False

                # Quit the task.
                if keyboard.is_pressed("q"):
                    return
                # Stop the trial and calculate the time.
                elif keyboard.is_pressed('s') and not new_trial:
                    belt_controller.stop_vibration()
                    print("stop")
                    end = time.time()
                    elapsed = end - begin
                    elapsed = elapsed
                    new_trial = True
                    print("Trial completed.")
                    print("Completion time is ", elapsed, "seconds")
                    obs_grasping.append([elapsed, num_instructions, "tactile"])
                    num_instructions = 0
                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('right'):
                    curr = "r"
                    print("right")
                    belt_controller.vibrate_at_angle(120, channel_index=0)
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('left'):
                    curr = "l"
                    belt_controller.vibrate_at_angle(45, channel_index=0)
                    print("left")
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('down'):
                    curr = "d"
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    belt_controller.vibrate_at_angle(60, channel_index=0)
                    print("down")
                    
                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('up'):
                    curr = "u"
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    belt_controller.vibrate_at_angle(90, channel_index=0)
                    print("up")
                    
                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('f'):
                    belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=None,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=9,
                        series_period=5000,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
                    curr = "f"
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    
                    break
                else:
                    break
            except ValueError:
                if keyboard.is_pressed('p'):
                    belt_controller.disconnect_belt()
                else:
                    break

    return 0

def collect_response():
    time_post_stim = time.time()
    response = ""
    while time.time() - time_post_stim < 3:
        if keyboard.is_pressed("left"):
            response = "left"
        if keyboard.is_pressed("right"):
            response = "right"
        if keyboard.is_pressed("up"):
            response = "up"
        if keyboard.is_pressed("down"):
            response = "down"
    if response == "":
        response = "no response"
    return response

def grasping_task_auditory():
    # Get paths to audio files.
    audio_right = Path().cwd() / "instruction_right.wav"
    audio_left = Path().cwd() / "instruction_left.wav"
    audio_up = Path().cwd() / "instruction_up.wav"
    audio_down = Path().cwd() / "instruction_down.wav"
    audio_forward = Path().cwd() / "instruction_forward.wav"

    curr = ""
    last = ""
    new_trial = True
    num_instructions = 0

    while True:
        if (keyboard.is_pressed("left") or keyboard.is_pressed("right") or keyboard.is_pressed("up") or keyboard.is_pressed("down")) and new_trial:
            begin = time.time()
            new_trial = False

        if keyboard.is_pressed("q"):
            pygame.mixer.music.stop()
            return

        if keyboard.is_pressed('s') and not new_trial:
            end = time.time()
            pygame.mixer.music.stop() 
            print("stop")
            elapsed = end - begin
            new_trial = True
            print("Trial completed.")
            print("Completion time is ", elapsed, "seconds")
            print("Number of instructions is ", num_instructions)
            obs_grasping.append([elapsed, num_instructions, "auditory"])
            num_instructions = 0
        
        elif keyboard.is_pressed('right'):
            curr = "r"
            if curr != last:
                pygame.mixer.music.load(audio_right)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr

        elif keyboard.is_pressed('left'):
            curr = "l"
            if curr != last:
                pygame.mixer.music.load(audio_left)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr
        
        elif keyboard.is_pressed('up'):
            curr = "u"
            if curr != last:
                pygame.mixer.music.load(audio_up)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr

        elif keyboard.is_pressed('down'):
            curr = "d"
            if curr != last:
                pygame.mixer.music.load(audio_down)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr
        
        elif keyboard.is_pressed('f'):
            curr = "f"
            if curr != last:
                pygame.mixer.music.load(audio_right)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr





if __name__ == "__main__":
    main()