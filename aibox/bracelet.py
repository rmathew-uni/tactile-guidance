# Some comments:

# issues.... perhaps every n frame this bbox_info should be wipped clean....
# or when the hand reaches an object.....
# empty it when it reaches the object or empty it within limited frame, like 20 or 50....

# now we have four numbers...which we need two of them the first two...
# write a function so it would outout left, right, up, down so the two number reach each other!!!!

# bbox_info ---- This is a list...each element is dictionary which had the key label and bbox
# we only need to accses the label that is person....and one other object
# then these two are needed for always....

# xywh --- xy is the center!
# simple version of bounding boxes

from pybelt.belt_controller import (BeltConnectionState, BeltController,
                                    BeltControllerDelegate, BeltMode,
                                    BeltOrientationType,
                                    BeltVibrationTimerOption)

from connect import interactive_belt_connect, setup_logger

class Delegate(BeltControllerDelegate):
    # Belt controller delegate
    pass

def connect_belt():
    setup_logger()

    # Interactive script to connect the belt
    interactive_belt_connect(belt_controller)
    if belt_controller.get_connection_state() != BeltConnectionState.CONNECTED:
        print("Connection failed.")
        return 0

    # Change belt mode to APP mode
    belt_controller.set_belt_mode(BeltMode.APP_MODE)

belt_controller_delegate = Delegate()
belt_controller = BeltController(belt_controller_delegate)

connect_belt()

vibration_intensity = 100

# bbox_info = [{"Label": "person", "Bbox": [949.0, 527.5, 532.0, 379.0]},
#              {"Label": "tv", "Bbox": [581.5, 670.5, 229.0, 97.0]},
#              {"Label": "banana", "Bbox": [2.0, 527.5, 532.0, 379.0]},
#              {"Label": "tv", "Bbox": [948.0, 527.5, 532.0, 3.0]},
#              {"Label": "person", "Bbox": [800.0, 527.5, 532.0, 4.0]}
#              ]
# print(bbox_info)

# find the index of person, and object
# Define the key and value
search_key_hand = "Label"
search_value_hand = "person"

search_key_obj = "Label"
search_value_obj = "tv"

def navigate_hand(bbox_info, search_key_obj, search_key_hand,hor_correct = False, ver_correct = False):
    # Using a loop to find the index
    #index_hand = None
    #index_obj = None

    horizontal, vertical = False, False

    max_hand_confidence = 0
    max_obj_confidence = 0

    print(bbox_info)

    # acquire the latest bbox information about hand and object
    '''
    for i, item in reversed(list(enumerate(bbox_info))):
        if item.get(search_key_hand) == search_value_hand:
            index_hand = i
        elif item.get(search_key_obj) == search_value_obj:
            index_obj = i
            break
    '''

    bbox_hand, bbox_obj = None, None

    for bbox in bbox_info:
        if bbox["class"] in search_key_hand and bbox["confidence"] > max_hand_confidence:
            bbox_hand = bbox.get("bbox")
            max_hand_confidence = bbox["confidence"]
        elif bbox["class"] == search_key_obj and bbox["confidence"] > max_obj_confidence:
            bbox_obj = bbox.get("bbox")
            max_obj_confidence = bbox["confidence"]

    if bbox_hand != None and bbox_obj == None and hor_correct and ver_correct:
        print("G R A S P !")
        belt_controller.send_pulse_command(
                        channel_index=0,
                        orientation_type=BeltOrientationType.ANGLE,
                        orientation=90,
                        intensity=vibration_intensity,
                        on_duration_ms=150,
                        pulse_period=500,
                        pulse_iterations=9,
                        series_period=5000,
                        series_iterations=1,
                        timer_option=BeltVibrationTimerOption.RESET_TIMER,
                        exclusive_channel=False,
                        clear_other_channels=False
                    )
        return True, True

    if bbox_hand == None or bbox_obj == None:
        belt_controller.stop_vibration()
        print('no find')
        return False, False

    #bbox_hand = bbox_info[index_hand].get("Bbox")
    #bbox_obj = bbox_info[index_obj].get("Bbox")

    x_center_hand, y_center_hand = bbox_hand[0], bbox_hand[1]
    x_center_obj, y_center_obj = bbox_obj[0], bbox_obj[1]


    # This Will be adjusted if within if-loop
    x_threshold = 100
    y_threshold = 100

    # Vertical movement
    if abs(y_center_hand - y_center_obj) > y_threshold:
        if y_center_hand < y_center_obj:
            belt_controller.vibrate_at_angle(60, channel_index=0, intensity=vibration_intensity)
            print('down')
        elif y_center_hand > y_center_obj:
            belt_controller.vibrate_at_angle(90, channel_index=0, intensity=vibration_intensity)
            print('up')
    else:
        vertical = True

    # Horizontal movement
    if abs(x_center_hand - x_center_obj) > x_threshold:
        if x_center_hand < x_center_obj:
            # Here we use the script for bracelet
            print('right')
            belt_controller.vibrate_at_angle(120, channel_index=0, intensity=vibration_intensity)
        elif x_center_hand > x_center_obj:
            print('left')
            belt_controller.vibrate_at_angle(45, channel_index=0, intensity=vibration_intensity)
    else:
        horizontal = True

    return horizontal, vertical