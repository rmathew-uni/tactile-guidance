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

# find the index of person, and object
# Define the key and value
search_key_hand = "Label"
search_value_hand = "person"

search_key_obj = "Label"
search_value_obj = "tv"

def navigate_hand(bbox_info, search_key_obj, search_value_obj, search_key_hand, search_value_hand, hor_correct = False, ver_correct = False):
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
        if bbox["label"] == search_key_hand and bbox["confidence"] > max_hand_confidence:
            bbox_hand = bbox.get("bbox")
            max_hand_confidence = bbox["confidence"]
        elif bbox["label"] == search_key_obj and bbox["confidence"] > max_obj_confidence:
            bbox_obj = bbox.get("bbox")
            max_obj_confidence = bbox["confidence"]

    if bbox_hand != None and bbox_obj == None and hor_correct and ver_correct:
        print("G R A S P !")
        return True, True

    if bbox_hand == None or bbox_obj == None:
        print("Hand or object not detected")
        return False, False

    #bbox_hand = bbox_info[index_hand].get("Bbox")
    #bbox_obj = bbox_info[index_obj].get("Bbox")

    # get the original locations of the center of the bbox for the hand
    x_center_hand, y_center_hand = bbox_hand[0], bbox_hand[1]
    
    # designate a target point in the bbox that is on the fingers (moved up from the center of the box)
    # the x center stays the same for this and we update the y coordinate of the center
    #y_center_hand = bbox_hand[1]-(bbox_hand[3]/4)

    x_center_obj, y_center_obj = bbox_obj[0], bbox_obj[1]


    # This Will be adjusted if within if-loop
    x_threshold = 10
    y_threshold = 10
    # Horizontal movement
    if abs(x_center_hand - x_center_obj) > x_threshold:
        if x_center_hand < x_center_obj:
            # Here we use the script for bracelet
            print("right")
        elif x_center_hand > x_center_obj:
            print("left")
    else:
        horizontal = True

    # Vertical movement
    if abs(y_center_hand - y_center_obj) > y_threshold:
        if y_center_hand < y_center_obj:
            print("down")
        elif y_center_hand > y_center_obj:
            print("up")
    else:
        vertical = True

    return horizontal, vertical