from pynput.mouse import Button, Listener

def on_click(x, y, button):
    # Check if the pressed button is the right mouse button
    if button == Button.left:
        print("Next slide please!")
        # Implement what happens after the clicker button is pressed
        # Modify this to the master.py script 

    # Check if the pressed button is the left mouse button to stop the listener
    if button == Button.right:
        return False

# Set up the listener for mouse events
with Listener(on_click=on_click) as listener:
    listener.join()


        






