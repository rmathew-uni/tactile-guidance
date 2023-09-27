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
bbox_info = [{"Label": "person", "Bbox": [949.0, 527.5, 532.0, 379.0]},
             {"Label": "tv", "Bbox": [581.5, 670.5, 229.0, 97.0]},
             {"Label": "banana", "Bbox": [2.0, 527.5, 532.0, 379.0]},
             {"Label": "tv", "Bbox": [948.0, 527.5, 532.0, 3.0]},
             {"Label": "person", "Bbox": [800.0, 527.5, 532.0, 4.0]}
             ]
print(bbox_info)

# find the index of person, and object
# Define the key and value
search_key_hand = "Label"
search_value_hand = "person"

search_key_obj = "Label"
search_value_obj = "tv"

# Using a loop to find the index
index_hand = None
index_obj = None
# acquire the latest bbox information about hand and object
for i, item in reversed(list(enumerate(bbox_info))):
    if item.get(search_key_hand) == search_value_hand:
        index_hand = i
    elif item.get(search_key_obj) == search_value_obj:
        index_obj = i
        break

bbox_hand = bbox_info[index_hand].get("Bbox")
bbox_obj = bbox_info[index_obj].get("Bbox")

x_center_hand, y_center_hand = bbox_hand[0], bbox_hand[1]
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

# Vertical movement
if abs(y_center_hand - y_center_obj) > y_threshold:
    if y_center_hand < y_center_obj:
        print("down")
    elif y_center_hand > y_center_obj:
        print("up")
