# region Setup

import time
from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption)
from auto_connect import interactive_belt_connect, setup_logger
import threading
import sys
from pynput.keyboard import Key, Listener
import numpy as np

# endregion

class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass


def connect_belt():
    setup_logger()

    belt_controller_delegate = Delegate()
    belt_controller = BeltController(belt_controller_delegate)

    # Interactive script to connect the belt
    interactive_belt_connect(belt_controller)
    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Connection failed.")
        global mock_belt
        #mock_belt = 1
        return False, belt_controller
    else:
        # Change belt mode to APP mode
        belt_controller.set_belt_mode(BeltMode.APP_MODE)
        return True, belt_controller


def abort(key):
    # Check if the pressed key is the left clicker key
    if key == Key.esc:
        sys.exit()


def on_click(key):
    # Check if the pressed key is the right clicker key
    if key == Key.enter:
        return False


def listener():
    # listen for clicker
    with Listener(on_press=abort) as listener:
        listener.join()


def start_listener():
    global termination_signal, one_round
    existing_thread = threading.enumerate()
    listener_thread = None

    for thread in existing_thread:
        if thread.name == 'clicker':
            listener_thread = thread
            termination_signal = False
            break
    
    if listener_thread is None:
        if one_round == 0:
            listener_thread = threading.Thread(target=listener, name='clicker')
            listener_thread.start()
            one_round += 1
        else:
            termination_signal = True
    
    return termination_signal


def choose_detection(bboxes, previous_bbox=None):
    # Hyperparameters
    track_id_weight = 1000
    exponential_weight = 2
    distance_weight = 100

    #print(f'\nPrevious BB: {previous_bbox}')

    candidates = []
    for bbox in bboxes: # x, y, w, h, id, cls, conf
        # bbox has to be within image dimensions
        if bbox[0] <= w and bbox[1] <= h:
            # confidence score
            confidence = bbox[6] # in [0,1]
            confidence_score = exponential_weight**confidence - 1 # exponential growth in [0,1], could also use np.exp() and normalize
            # tracking score
            current_track_id = bbox[4]
            previous_track_id = previous_bbox[4] if previous_bbox is not None else -1
            track_id_score = track_id_weight if current_track_id == previous_track_id else 1 # 1|ꝏ
            # distance score
            if previous_bbox is None:
                distance = None
                distance_inverted = 1
            else:
                current_location = bbox[:2]
                previous_location = previous_bbox[:2]
                distance = np.linalg.norm(current_location - previous_location)
                distance_inverted = 1 / distance if distance >= 1 else distance_weight

            # total score
            score = track_id_score * confidence_score * distance_inverted
            #print(f'Current BB: {bbox}')
            #print(f'TrackID = {current_track_id}, confidence = {confidence}, distance = {distance}')
            #print(f'Score {score} = {track_id_score} * {confidence_score} * {distance_inverted}\n')

            # Possible scores:
            # ꝏ -- same trackingID
            # 100 -- different trackingID, matching BBs (max. 1px deviation), conf=1
            # [0,1] -- different trackingID, BBs distance in [1., sqrt(w^2*h^2)], conf=1
            candidates.append(score)
        else:
            candidates.append(0)

    true_detection = bboxes[np.argmax(candidates)] if len(candidates) else None

    return true_detection


# Threading vars
termination_signal = False
one_round = 0

# Navigation vars
prev_hand = None
prev_target = None
w,h = 1920, 1080

def navigate_hand(
        belt_controller,
        bboxes, 
        search_key_obj: str, 
        search_key_hand: list,
        hor_correct: bool = False, 
        ver_correct: bool = False, 
        grasp: bool = False, 
        obj_seen_prev: bool = False, 
        search: bool = False, 
        count_searching: int = 0, 
        count_see_object: int = 0, 
        jitter_guard: int = 0, 
        navigating: bool = False
        ):
    
    '''
    Function that navigates the hand to the target object. Handles cases when either hand or target is not detected
    Input:
    • bboxes - list containing following information about each prediction: 0-3: bbox xywh, 4: trackID, 5: class, 6: confidence
    • search_key_obj - integer representing target object class
    • search_key_hand - list of integers containing hand detection classes used for navigation
    • hor_correct - boolean representing whether hand and object are assumed to be aligned horizontally; by default False
    • ver_correct - boolean representing whether hand and object are assumed to be aligned vertically; by default False
    • grasp - boolean representing whether grasp command has been sent; by default False
    • x_threshold - 
    • y_threshold - 
    
    Output:
    • horizontal - boolean representing whether hand and object are aligned horizontally after execution of the function; by default False
    • vertical - boolean representing whether hand and object are aligned vertically after execution of the function; by default False
    • grasp - boolean representing whether grasp command has been sent; by default False
    • check
    • check_dur
    '''

    global termination_signal
    global prev_hand, prev_target

    # Navigation vars
    vibration_intensity = 100
    min_hand_confidence = 0.5
    min_obj_confidence = 0.5
    hand, target = None, None
    horizontal, vertical = False, False
    w,h = 1920, 1080

    if belt_controller:
        termination_signal = start_listener()

    if termination_signal:
        print('Manual Abort')
        belt_controller.stop_vibration()
        sys.exit()

    # Search for object and hand with the highest prediction confidence
    # Filter for hand detections
    bboxes_hands = [detection for detection in bboxes if detection[5] in search_key_hand]
    hand = choose_detection(bboxes_hands, prev_hand)
    prev_hand = hand

    # Filter for target detections
    bboxes_objects = [detection for detection in bboxes if detection[5] == search_key_obj]
    target = choose_detection(bboxes_objects, prev_target)
    prev_target = target

    # Getting horizontal and vertical position of the bounding box around target object and hand
    if hand is not None:
        x_center_hand, y_center_hand = hand[0], hand[1]
        # move the y_center of the hand in the direction of the fingertips to help avoid occlusions (testing)
        hand_upper_bound = y_center_hand - hand[3]//2
        hand_lower_bound = y_center_hand + hand[3]//2

    if target is not None:
        target_width = target[2]
        target_height = target[3]
        target_left_bound = target[0] - target_width//2
        target_right_bound = target[0] + target_width//2
        target_lower_bound = target[1] + target_height//2
        target_upper_bound = target[1] - target_height//2
 

    # 1. Grasping: Hand is detected and horizontally and vertically aligned with target --> send grasp (target might be occluded in frame)
    if hand is not None and hor_correct and ver_correct:
        obj_seen_prev = False
        search = False
        count_searching = 0
        count_see_object = 0
        jitter_guard = 0
        navigating = 0

        if belt_controller:
            belt_controller.stop_vibration()
            belt_controller.send_pulse_command(
                            channel_index=0,
                            orientation_type=BeltOrientationType.ANGLE,
                            orientation=90,
                            intensity=vibration_intensity,
                            on_duration_ms=150,
                            pulse_period=500,
                            pulse_iterations=5,
                            series_period=5000,
                            series_iterations=1,
                            timer_option=BeltVibrationTimerOption.RESET_TIMER,
                            exclusive_channel=False,
                            clear_other_channels=False
                        )

        print("G R A S P !")
        
        # End guidance RT measure
        print('Please use the clicker to indicate you have grasped the object.')

        # listen for clicker
        with Listener(on_press=on_click) as listener:
            # End trial time measure
            listener.join()

        grasp = True

        return horizontal, vertical, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating


    # 2. Guidance: If the camera can see both hand and object but not yet aligned, navigate the hand to the object, horizontal first
    if hand is not None and target is not None:
        obj_seen_prev = False
        search = False
        count_searching = 0
        count_see_object = 0
        jitter_guard = 0
        threshold = 20 * target[2] / hand[2]

        # Start guidance RT measure

        # Horizontal movement logic
        # Centers of the hand and object bounding boxes further away than x_threshold - move hand horizontally
        horizontal = False
        if x_center_hand < target_left_bound:
            print('right')
            if belt_controller:
                belt_controller.vibrate_at_angle(120, channel_index=0, intensity=vibration_intensity)
            navigating = True
        elif x_center_hand > target_right_bound:
            print('left')
            if belt_controller:
                belt_controller.vibrate_at_angle(45, channel_index=0, intensity=vibration_intensity)
            navigating = True
        else:
            horizontal = True

        # Vertical movement logic
        # Centers of the hand and object bounding boxes further away than y_threshold - move hand vertically
        if horizontal == True:
            vertical = False
            if hand_lower_bound < target_upper_bound: # - threshold: # dynamic grasp triggering
                print('down')
                if belt_controller:
                    belt_controller.vibrate_at_angle(60, channel_index=0, intensity=vibration_intensity)
                navigating = True
            elif hand_upper_bound > target_lower_bound: # + threshold:
                print('up')
                if belt_controller:
                    belt_controller.vibrate_at_angle(90, channel_index=0, intensity=vibration_intensity)
                navigating = True
            else:
                vertical = True

        return horizontal, vertical, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating


    # 3. Lost target: If the camera cannot see the hand or the object, tell them they need to move around
    if target is None and grasp == False:
        if obj_seen_prev == True:
            jitter_guard = 0 
            obj_seen_prev = False

        #print("Lost target from the field of view.")
        
        jitter_guard += 1
        if jitter_guard >= 40:
            count_see_object = 0 
            navigating = False

            if belt_controller:
                if search == False:
                        belt_controller.stop_vibration()
                        # left
                        belt_controller.send_pulse_command(
                                    channel_index=0,
                                    orientation_type=BeltOrientationType.ANGLE,
                                    orientation=45,
                                    intensity=vibration_intensity,
                                    on_duration_ms=100,
                                    pulse_period=500,
                                    pulse_iterations=3,
                                    series_period=5000,
                                    series_iterations=1,
                                    timer_option=BeltVibrationTimerOption.RESET_TIMER,
                                    exclusive_channel=False,
                                    clear_other_channels=False
                                )
                        search = True

            count_searching += 1
            if count_searching >= 150:
                search = False
                count_searching = 0
            
        return horizontal, vertical, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating


    # 4. Lost hand: If the camera cannot see the hand but the object is visible, tell them to move the hand around
    if target is not None:

        if search == True:
            jitter_guard = 0
            search = False

        jitter_guard += 1
        if jitter_guard >= 40: 
            navigating = False
            count_searching = 0

            #print("Lost hand from the field of view.")
            
            if obj_seen_prev == False:
                if belt_controller:
                    belt_controller.stop_vibration()
                    #down
                    belt_controller.send_pulse_command(
                                channel_index=0,
                                orientation_type=BeltOrientationType.ANGLE,
                                orientation=120,
                                intensity=vibration_intensity,
                                on_duration_ms=100,
                                pulse_period=500,
                                pulse_iterations=3,
                                series_period=5000,
                                series_iterations=1,
                                timer_option=BeltVibrationTimerOption.RESET_TIMER,
                                exclusive_channel=False,
                                clear_other_channels=False
                            )
                obj_seen_prev = True

            count_see_object += 1
            if count_see_object >= 150:
                obj_seen_prev = False
                count_see_object = 0
        
        return horizontal, vertical, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating
    
    else:
        print('Condition not covered by logic. Maintaining variables and standing by.')
        grasp = False
        return horizontal, vertical, grasp, obj_seen_prev, search, count_searching, count_see_object, jitter_guard, navigating