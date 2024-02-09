from pynput.keyboard import Key, Listener

def on_press(key):
    # Check if the pressed key is the right arrow key
    if key == Key.page_down:
        print("Next slide please!")
        # Implement what happens after the clicker button is pressed
        # Modify this to the master.py script 

def on_release(key):
    # Check if the pressed key is the escape key
    if key == Key.esc:
        # Stop the listener
        return False

# Set up the listener for keyboard events
with Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

        






