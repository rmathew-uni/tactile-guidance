#from operator import le
#from sre_constants import SUCCESS
import csv
import random
import time
from pathlib import Path
from random import randrange

import keyboard
import pygame
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption)

from connect import interactive_belt_connect, setup_logger


class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass

pygame.init()
random.seed()

# Global variables
participantID = 0
belt_controller_delegate = Delegate()
belt_controller = BeltController(belt_controller_delegate)
obs_localization = []
block_localization = 1
presented_stimuli_localization = []
obs_grasping = [["time", "num_instructions", "condition", "block", "success"]]
block_grasping = 1


def main():
    print("Welcome to the \"Augmenting functional vision using tactile guidance\" experiment!")
    print("Please connect the tactile bracelet and press Enter to continue.")

    # Wait for user to continue.
    while True:
        if keyboard.is_pressed("enter"):
            break

    belt_controller = connect_belt()

    # Generate a new Participant ID or enter one.
    ui = get_input("Genereate new participant ID? [y,n]", range_=["y","n"])

    if ui == "y":
        participantID = generate_participantID()
        print("Participant ID: " + participantID)  
    else:
        participantID = get_input("Please enter a participant ID: ")
        print("Participant ID: " +  participantID)

    print("Coinflip:" + str(random.sample([0,1], 1)))
    print("Press 1 to start the localization task. Press 2 to start the grasping task. Press 3 to start calibration.")

    # Select localization or grasping task.
    while True:
        user_in = get_input("Localization, grasping or calibration? [1,2,3]", type_=int, min_=1, max_=3)

        # Start localization task
        if user_in == 1:
            localization_task()
            print(obs_localization)
            calc_accuracy()
            write_to_csv(participantID, obs_localization, "localization")
            write_to_csv(participantID, presented_stimuli_localization, "stimuli")

        # Start grasping task
        if user_in == 2:
            user_in = get_input("Tactile or auditory condition? [1,2]", type_=int, min_=1, max_=2)
            if user_in == 1:
                grasping_task("tactile")
                #grasping_task_tactile()
                print(obs_grasping)
                write_to_csv(participantID, obs_grasping, "grasping")
            elif user_in == 2:
                    #grasping_task_auditory()
                    grasping_task("auditory")
                    print(obs_grasping)
                    write_to_csv(participantID, obs_grasping, "grasping")

        if user_in == 3:
            calibrate_motors()
def write_to_csv(id, observations, task):

    #if file exists read rows into list so that new obs can be added to old
    id = id
    obs = observations
    if task == "localization":
        filepath = str(id + "_localization" + ".csv")

    elif task == "grasping":
        filepath = str(id + "_grasping" + ".csv")
    
    elif task == "stimuli":
        filepath = str(id + "_stimuli" + ".csv")

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
        print("Q. to quit.")
        print("0. Stop vibration.")
        print("1. Example stimuli.")
        print("2. Start block.")
        
        global block_localization
       
        while True:
            user_in = get_input("[Q,0,1,2]: ", type_=str, range_=["0", "1", "2", "Q","q"])
            
            # Start a block.
            if user_in == "2":
                stop = False
                while not stop:
                    print("Current block is number " + str(block_localization))
                    stimuli = [45,60,90,120] # List of orientations

                    random.shuffle(stimuli) # Shuffle orientations
                    obs_localization.append([])
                    trans_stimuli = [] # List with stimuli translated from orientation to direction.
                    # Translate stimuli from orientation to direction.
                    for stimulus in stimuli:
                        if stimulus == 45: trans_stimuli.append("left")
                        elif stimulus == 60: trans_stimuli.append("down")
                        elif stimulus ==  90: trans_stimuli.append("up")
                        elif stimulus == 120: trans_stimuli.append("right")
                    presented_stimuli_localization.append(trans_stimuli)
                
                    print(stimuli)

                    time.sleep(3)
                    for stimulus in stimuli:
                        print(stimulus)
                        belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=stimulus,
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
                        obs_localization[block_localization-1].append(collect_response())
                        #time.sleep(3)
                    
                    user_in = get_input("Block " + str(block_localization) + " completed. Continue?[y,n]", type_=str, range_=["y","n"])
                    #user_in = input("Block " + str(block_localization) + " completed. Continue?[y,n]")
                    block_localization += 1
                    if user_in == "n": stop = True 
                print(presented_stimuli_localization)
                break

            # Present example stimuli.
            elif user_in == "1":
                #Up
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
                time.sleep(3)
                #Right
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
                time.sleep(3)
                #Down
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
                time.sleep(3)
                #Left
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
                time.sleep(3)
                break
            
            # Quit the task.
            if user_in.lower() == "q":
                return

def grasping_task(condition):

    if condition == "auditory":
        # Get paths to audio files.
        audio_right = Path().cwd() / "instruction_right.wav"
        audio_left = Path().cwd() / "instruction_left.wav"
        audio_up = Path().cwd() / "instruction_up.wav"
        audio_down = Path().cwd() / "instruction_down.wav"
        audio_forward = Path().cwd() / "instruction_forward.wav"
    
    new_trial = True
    num_instructions = 0
    time_limit = 30
    last = ""
    time_limit = 30
    begin = 0
    target_idx = 0
    rep_idx = 0
    target_list = list(range(1,10))
    random.shuffle(target_list)
    rep_list = []

    # Present example stimuli.
    user_in = get_input("Present example stimuli? [y,n]", range_=("y","n"))
    if user_in == "y":
        if condition == "tactile":
            present_example_tactile()
        elif condition == "auditory":
            present_example_auditory()
    
    # Get block number or quit.
    if condition == "tactile":
        block_tactile = get_input("Enter a block number or press 0 to quit the task.", type_=int, min_=0)
        if block_tactile == 0: # Quit if user enters 0.
            return
    if condition == "auditory":
        block_auditory = get_input("Enter a block number or press 0 to quit the task.", type_=int, min_=0)
        if block_auditory == 0: # Quit if user enters 0.
            return
    
    # Print targets.
    print("Target order: " + str(target_list))
    print("Target: " + str(target_list[target_idx]))

    while True:
        # Check for start of trial.
        if (keyboard.is_pressed("left") or keyboard.is_pressed("right") or keyboard.is_pressed("up") or keyboard.is_pressed("down")) and new_trial and not keyboard.is_pressed("s"):
                    begin = time.perf_counter()
                    #print(begin)
                    #num_instructions += 1
                    new_trial = False
        
        # Check time limit.
        elif(begin != 0):
            if(time.perf_counter() - begin > time_limit) and not new_trial:
                if condition == "tactile":
                    belt_controller.stop_vibration()
                    obs_grasping.append([time_limit, num_instructions, target_idx, block_tactile, "tactile", "fail"])
                elif condition == "auditory":
                    pygame.mixer.music.stop()
                    obs_grasping.append([time_limit, num_instructions, target_idx, block_auditory, "auditory", "fail"])
                print("stop")
                print("Time limit reached.")
                num_instructions = 0
                new_trial = True
                #break

        # Quit the task.
        if keyboard.is_pressed("q"):
            if condition == "tactile":
                belt_controller.stop_vibration()
            elif condition == "auditory":
                pygame.mixer.music.stop()
            return
        
        # Stop the trial and calculate the time.
        elif keyboard.is_pressed('s') and not new_trial:
            end = time.perf_counter()
            elapsed = end - begin # Calculate trial time.
            if condition == "tactile":
                belt_controller.stop_vibration()
            elif condition == "auditory":
                pygame.mixer.music.stop()
            print("stop")
            new_trial = True
            print("Trial completed.")
            print("Completion time is ", elapsed, "seconds.")
            result = get_input("Press s for success, d for fail or f for experimenter fail.[s,d,f]", type_=str, range_=("s","d","f"))
            if result == "s": result = "success"
            elif result == "d": result = "fail"
            # Experimenter fail, append target to repetion list.
            else: 
                result = "exFail"
                if target_idx < 9: # Check if all targets have been recorded before.
                    rep_list.append(target_list[target_idx]) # Append target where experimenter failed to repetition list.
                else: # If all targets have been recorded.
                    try:
                        rep_list.append(rep_list[rep_idx-1]) 
                    except: print("out of index")
            # No experimenter fail.
            if target_idx < 9: 
                if condition == "tactile": 
                    obs_grasping.append([elapsed, num_instructions, target_list[target_idx], block_tactile, "tactile", result])
                elif condition == "auditory":
                    obs_grasping.append([elapsed, num_instructions, target_list[target_idx], block_auditory, "auditory", result])
            elif len(rep_list) > rep_idx-1:
                if condition == "tactile":
                    obs_grasping.append([elapsed, num_instructions, "rep_" + str(rep_list[rep_idx-1]), block_grasping, "tactile", result])
                elif condition == "auditory":
                    obs_grasping.append([elapsed, num_instructions, "rep_" + str(rep_list[rep_idx-1]), block_auditory, "auditory", result])
            num_instructions = 0
            last = ""
            target_idx += 1
            if target_idx > 8: # Check if all targets have been recorded.
                if len(rep_list) != 0: # Check if targets must be repeated.
                    if len(rep_list) > rep_idx:
                        print("Repetition target:" + str(rep_list[rep_idx])) 
                        rep_idx += 1
                        
                    else: print("Block finished!")
                else: print("Block finished!")
            else: print(target_list[target_idx]) # Print target.                    
        
            #time.sleep(0.5)
            #break

        elif keyboard.is_pressed('right') and not new_trial and last != "r":
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(120, channel_index=0)
                elif condition == "auditory":
                    pygame.mixer.music.load(audio_right)
                    pygame.mixer.music.play(-1)
                num_instructions += 1
                last = "r"

        elif keyboard.is_pressed('left') and not new_trial and last != "l":
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(45, channel_index=0)
                elif condition == "auditory":
                    pygame.mixer.music.load(audio_left)
                    pygame.mixer.music.play(-1)
                num_instructions += 1
                last = "l"

        elif keyboard.is_pressed('down') and not new_trial and last != "d":
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(60, channel_index=0)
                elif condition == "auditory":
                    pygame.mixer.music.load(audio_down)
                    pygame.mixer.music.play(-1)
                num_instructions += 1
                last = "d"

        elif keyboard.is_pressed('up') and not new_trial and last != "u":
                if condition == "tactile":
                    belt_controller.vibrate_at_angle(90, channel_index=0)
                elif condition == "auditory":
                    pygame.mixer.music.load(audio_up)
                    pygame.mixer.music.play(-1)
                num_instructions += 1
                last = "u"

        elif keyboard.is_pressed('f') and not new_trial and last != "f":
                if condition == "tactile":
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
                elif condition == "auditory":
                    pygame.mixer.music.load(audio_forward)
                    pygame.mixer.music.play(-1)
                num_instructions += 1
                last = "f"

    
        



def grasping_task_tactile():
    new_trial = True
    num_instructions = 0
    curr = ""
    last = ""
    time_limit = 30
    begin = 0
    first = True
    target_idx = 0
    rep_idx = 0
    l = list(range(1,10))
    random.shuffle(l)
    rep_list = list() # List to store targets that must be repeated
    

    user_in = get_input("Present example stimuli? [y,n]", range_=("y","n"))
    if user_in == "y":
        present_example_tactile()
    
    block_grasping = get_input("Enter a block number or press 0 to quit the task.", type_=int, min_=0)
    if block_grasping == 0:
        return
    print("Target order: " + str(l))
    print("Target: " + str(l[target_idx]))
       
    while belt_controller.get_connection_state() == BeltConnectionState.CONNECTED:

       

        while True:
            try:
                #print(num_instructions)
                # Check if its the first instruction and if it is log the start time.
                
               
                
                if (keyboard.is_pressed("left") or keyboard.is_pressed("right") or keyboard.is_pressed("up") or keyboard.is_pressed("down")) and new_trial and not keyboard.is_pressed("s"):
                    begin = time.perf_counter()
                    print(begin)
                    #num_instructions += 1
                    new_trial = False
                
                # Check time limit.
                elif(begin != 0):
                    if(time.perf_counter() - begin > time_limit) and not new_trial:
                        belt_controller.stop_vibration()
                        print("stop")
                        print("Time limit reached.")
                        obs_grasping.append([time_limit, num_instructions, target_idx, block_grasping, "tactile", "fail"])
                        num_instructions = 0
                        new_trial = True
                        #break

                # Quit the task.
                if keyboard.is_pressed("q"):
                    belt_controller.stop_vibration()
                    return

                # Stop the trial and calculate the time.
                elif keyboard.is_pressed('s') and not new_trial:
                    belt_controller.stop_vibration()
                    print("stop")
                    end = time.perf_counter()
                    elapsed = end - begin
                    #elapsed = elapsed
                    new_trial = True
                    print("Trial completed.")
                    print("Completion time is ", elapsed, "seconds.")
                    result = get_input("Press s for success, d for fail or f for experimenter fail.[s,d,f]", type_=str, range_=("s","d","f"))
                    """ result = ""
                    while result not in ["s","d","f"]:
                        result = input("Press s for success, d for fail or f for experimenter fail.") """
                    if result == "s": result = "success"
                    elif result == "d": result = "fail"
                    else: 
                        result = "exFail"
                        if target_idx < 9:
                            rep_list.append(l[target_idx]) # Append target where experimenter failed to repetition list
                        else: 
                            try:
                                rep_list.append(rep_list[rep_idx-1])
                            except: print("out of index")
                    if target_idx < 9: 
                        obs_grasping.append([elapsed, num_instructions, l[target_idx], block_grasping, "tactile", result])
                    elif len(rep_list) > rep_idx-1:
                        obs_grasping.append([elapsed, num_instructions, "rep_" + str(rep_list[rep_idx-1]), block_grasping, "tactile", result])
                    num_instructions = 0
                    last = ""
                    
                    target_idx += 1
                    if target_idx > 8: # Check if all targets have been recorded.
                        if len(rep_list) != 0: # Check if targets must be repeated.
                            if len(rep_list) > rep_idx:
                                print("Repetition target:" + str(rep_list[rep_idx])) 
                                rep_idx += 1
                                
                            else: print("Block finished!")
                        else: print("Block finished!")
                    else: print(l[target_idx]) # Print target.                    
                
                    #time.sleep(0.5)
                    break
                # Stop a failed trial and calc time
                """  elif keyboard.is_pressed('d') and not new_trial:
                    belt_controller.stop_vibration()
                    print("stop")
                    end = time.perf_counter()
                    elapsed = end - begin
                    elapsed = elapsed
                    new_trial = True
                    print("Trial completed and failed.")
                    print("Completion time is ", elapsed, "seconds")
                    obs_grasping.append([elapsed, num_instructions, "tactile", "false"])
                    num_instructions = 0
                    #time.sleep(0.5)
                    break """
                if keyboard.is_pressed('right') and not new_trial:
                    curr = "r"
                    #print("right")
                    belt_controller.vibrate_at_angle(120, channel_index=0)
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('left') and not new_trial:
                    curr = "l"
                    belt_controller.vibrate_at_angle(45, channel_index=0)
                   # print("left")
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('down') and not new_trial:
                    curr = "d"
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    belt_controller.vibrate_at_angle(60, channel_index=0)
                   # print("down")

                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('up') and not new_trial:
                    curr = "u"
                    if curr != last:
                        last = curr
                        num_instructions += 1
                    belt_controller.vibrate_at_angle(90, channel_index=0)
                   # print("up")

                    #time.sleep(0.5)
                    break
                elif keyboard.is_pressed('f') and not new_trial:
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
    time_post_stim = time.perf_counter()
    response = ""
    while time.perf_counter() - time_post_stim < 3:
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
    time_limit = 30
    begin = 0

    target_idx = 0
    rep_idx = 0
    target_list = list(range(1,10))
    random.shuffle(target_list)
    rep_list = list() # Store targets that must be repeated.

    user_in = input("Present example stimuli? [y,n]")

    if user_in == "y":
        present_example_auditory()
        print("Starting trials...")
    
    block_auditory = input("Enter Block number: ")
    print("Target order: " + str(target_list)) # Print target order.
    print("Target: " +  str(target_list[target_idx])) # Print first target.

    while True:
        if (keyboard.is_pressed("left") or keyboard.is_pressed("right") or keyboard.is_pressed("up") or keyboard.is_pressed("down")) and new_trial and not keyboard.is_pressed("s") and not keyboard.is_pressed("d"):
            begin = time.perf_counter()
            last = ""
            new_trial = False
        
        elif(begin != 0):
            if(time.perf_counter() - begin > time_limit) and not new_trial:
                        pygame.mixer.music.stop()
                        print("stop")
                        new_trial = True
                        print("Time limit reached.")
                        obs_grasping.append([time_limit, num_instructions, "auditory", "false"])
                        num_instructions = 0
                        

        if keyboard.is_pressed("q"):
            pygame.mixer.music.stop()
            return

        if keyboard.is_pressed('s') and not new_trial:
            end = time.perf_counter()
            pygame.mixer.music.stop()
            print("stop")
            elapsed = end - begin
            new_trial = True
            print("Trial completed.")
            print("Completion time is ", elapsed, "seconds")
            print("Number of instructions is ", num_instructions)
            result = ""
            while result not in ["s","d","f"]:
                result = input("Press s for success, d for fail or f for experimenter fail.")
            if result == "s": result = "success"
            elif result == "d": result = "fail"
            else: 
                result = "exFail"
                if target_idx < 9:
                    rep_list.append(target_list[target_idx]) # Append target where experimenter failed to repetition list
                else:
                    try:
                        rep_list.append(rep_list[rep_idx-1])
                    except: print("Out of Index")
            if target_idx < 9:
                obs_grasping.append([elapsed, num_instructions, target_list[target_idx], block_auditory, "auditory", result])
            elif len(rep_list) > rep_idx-1:
                obs_grasping.append([elapsed, num_instructions, "rep_" + str(rep_list[rep_idx-1]), block_grasping, "tactile", result])
            num_instructions = 0
            last = ""

            target_idx += 1
            if target_idx > 8: # Check if all targets have been recorded.
                if len(rep_list) != 0: # Check if targets must be repeated.
                    if len(rep_list) > rep_idx:
                        print("Repetition target:" + str(rep_list[rep_idx])) 
                        rep_idx += 1

                    else: print("Block finished!")
                else: print("Block finished!")
            else: print(target_list[target_idx]) # Print target.
            
            #obs_grasping.append([elapsed, num_instructions, "auditory", result])
            #num_instructions = 0
        # Failed trial
        """ if keyboard.is_pressed('d') and not new_trial:
            end = time.perf_counter()
            pygame.mixer.music.stop()
            print("stop")
            elapsed = end - begin
            new_trial = True
            print("Trial completed and failed.")
            print("Completion time is ", elapsed, "seconds")
            print("Number of instructions is ", num_instructions)
            obs_grasping.append([elapsed, num_instructions, "auditory", "false"])
            num_instructions = 0 """

        if keyboard.is_pressed('right') and not new_trial:
            curr = "r"
            if curr != last:
                pygame.mixer.music.load(audio_right)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr

        elif keyboard.is_pressed('left') and not new_trial:
            curr = "l"
            if curr != last:
                pygame.mixer.music.load(audio_left)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr

        elif keyboard.is_pressed('up') and not new_trial:
            curr = "u"
            if curr != last:
                pygame.mixer.music.load(audio_up)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr

        elif keyboard.is_pressed('down') and not new_trial:
            curr = "d"
            if curr != last:
                pygame.mixer.music.load(audio_down)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr

        elif keyboard.is_pressed('f') and not new_trial:
            curr = "f"
            if curr != last:
                pygame.mixer.music.load(audio_forward)
                pygame.mixer.music.play(-1)
                num_instructions += 1
                last = curr

def present_example_tactile():

    while True:
        if keyboard.is_pressed('s'):
            belt_controller.stop_vibration()
        if keyboard.is_pressed('right'):
            belt_controller.vibrate_at_angle(120, channel_index=0)
        if keyboard.is_pressed('up'):
            belt_controller.vibrate_at_angle(90, channel_index=0)
        if keyboard.is_pressed('down'):
            belt_controller.vibrate_at_angle(60, channel_index=0)
        if keyboard.is_pressed('left'):
            belt_controller.vibrate_at_angle(45, channel_index=0)
        if keyboard.is_pressed('f'):
            belt_controller.send_pulse_command(
                channel_index=0,
                orientation_type=BeltOrientationType.ANGLE,
                orientation=90,
                intensity=None,
                on_duration_ms=150,
                pulse_period=500,
                pulse_iterations=9,
                series_period=5000,
                series_iterations=3,
                timer_option=BeltVibrationTimerOption.RESET_TIMER,
                exclusive_channel=False,
                clear_other_channels=False
                )
        if keyboard.is_pressed("q"):
            belt_controller.stop_vibration()
            break
            
def present_example_auditory():
    

    audio_right = Path().cwd() / "instruction_right.wav"
    audio_left = Path().cwd() / "instruction_left.wav"
    audio_up = Path().cwd() / "instruction_up.wav"
    audio_down = Path().cwd() / "instruction_down.wav"
    audio_forward = Path().cwd() / "instruction_forward.wav"

    curr = ""
    last = ""

    while True:

        if keyboard.is_pressed("q"):
            pygame.mixer.music.stop()
            return
        if keyboard.is_pressed('s'):
            pygame.mixer.music.stop()
        elif keyboard.is_pressed('right'):
            curr = "r"
            if curr != last:
                pygame.mixer.music.load(audio_right)
                pygame.mixer.music.play(-1)
                last = curr
        elif keyboard.is_pressed('left'):
            curr = "l"
            if curr != last:
                pygame.mixer.music.load(audio_left)
                pygame.mixer.music.play(-1)
                last = curr
        elif keyboard.is_pressed('up'):
            curr = "u"
            if curr != last:
                pygame.mixer.music.load(audio_up)
                pygame.mixer.music.play(-1)
                last = curr
        elif keyboard.is_pressed('down'):
            curr = "d"
            if curr != last:
                pygame.mixer.music.load(audio_down)
                pygame.mixer.music.play(-1)
                last = curr
        elif keyboard.is_pressed('f'):
            curr = "f"
            if curr != last:
                pygame.mixer.music.load(audio_forward)
                pygame.mixer.music.play(-1)
                last = curr

def calibrate_motors():
    angle = 0
    motor = 0
    intensities = list([50,50,50,50])
    print("ayy")
    while True:
        print(intensities)
        if keyboard.is_pressed("q"):
            print(intensities)
            return
        if keyboard.is_pressed("0"):
            angle = 90
            motor = 0
        elif keyboard.is_pressed("1"):
            angle = 120
            motor = 1
        elif keyboard.is_pressed("2"):
            angle = 60
            motor = 2
        elif keyboard.is_pressed("3"):
            angle = 45
            motor = 3
        
        if keyboard.is_pressed("up"):
            if intensities[motor] <= 90:
                intensities[motor] += 10
        if keyboard.is_pressed("down"):
            test = intensities[motor]
            if intensities[motor] >= 10:
                intensities[motor] = intensities[motor] - 10

        belt_controller.vibrate_at_angle(angle, channel_index=0, intensity=intensities[motor])
        
        time.sleep(2)
        
def generate_participantID():
    participantID = str(str(randrange(10)) + str(randrange(10)) + str(randrange(10)) + str(randrange(10)))
    return participantID

# Function to get user input taken from stack overflow
def get_input(prompt, type_=None, min_=None, max_=None, range_=None):
    if min_ is not None and max_ is not None and max_ < min_:
        raise ValueError("min mus be less than or equal to max.")
    while True:
        ui = input(prompt)
        if type_ is not None:
            try:
                ui = type_(ui)
            except ValueError:
                print("Input type must be {0}.".format(type_.__name__))
                continue
        if max_ is not None and ui > max_:
            print("Input must be less than or equal to {0}.".format(max_))
        elif min_ is not None and ui < min_:
            print("Input must be greater than or equal to {0}.".format(min_))
        elif range_ is not None and ui not in range_:
            if isinstance(range_, range):
                template = "Input must be between {0.start} and {0.stop}."
                print(template.format(range_))
            else:
                template = "Input must be {0}."
                if len(range_) == 1:
                    print(template.format(*range_))
                else: 
                    expected = " or ".join((
                        ", ".join(str(x) for x in range_[:-1]),
                        str(range_[-1])
                    ))
                    print(template.format(expected))
        else:
            return ui

def calc_accuracy():
    correct = 0
    num_blocks = 0
    mean_correct = 0
    
    if len(obs_localization) >= 3: # Check if at least 3 blocks were recorded
        for x in range(-3,0): # Go through the last 3 blocks
            for i in range(len(obs_localization[x])):    
                if obs_localization[x][i] == presented_stimuli_localization[x][i]:
                    correct += 1 
            num_blocks += 1

        if num_blocks > 0:
            mean_correct = correct / (num_blocks * len(obs_localization[0]))
    
        print(str(mean_correct) + "in " + str(num_blocks) + " blocks")


if __name__ == "__main__":
    main()

