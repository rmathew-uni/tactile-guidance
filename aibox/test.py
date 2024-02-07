from pynput.mouse import Listener

def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at (x, y): {x}, {y}")
        return False

with Listener(on_click=on_click) as mouse_listener:
    mouse_listener.join()