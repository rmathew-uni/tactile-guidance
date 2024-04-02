# imports

# load networks 

# load camera

# define object being targeted
# define dominant hand

# master loop per frame from camera

# loop for the frames from the camera 

  # every 15 frames (0.5 second), grab the current frame and feed it to the two networks 

    # feed image into object network (ON) to find target object location 
    # feed image into hand network (HN) to find dominant hand location

  # see if horizontal movement needed

    # if HN x is more than 30% different from ON x, move HN left if HN x is larger and move HN right if HN x is lower

  # see if vertical movement needed 

    # if HN y is more than 30% different from ON y, move HN down if HN y is larger and move HN up if HN y is lower

  # if neither needed

    # send command to grasp

