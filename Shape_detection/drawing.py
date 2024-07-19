import cv2
import numpy as np
import pandas as pd

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

# Function to calculate the minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    minimum = 10000
    cname = ""
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

# Function to get x, y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = frame[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# Initialize camera
cap = cv2.VideoCapture(0)

cv2.namedWindow('Live Color Detection')
cv2.setMouseCallback('Live Color Detection', draw_function)

clicked = False
r = g = b = xpos = ypos = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if clicked:
        # Draw rectangle and display the color name and RGB values
        cv2.rectangle(frame, (20, 20), (750, 60), (b, g, r), -1)
        color_name = get_color_name(r, g, b)
        text = f"{color_name} R={r} G={g} B={b}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if r + g + b >= 600:
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    cv2.imshow('Live Color Detection', frame)
    
    # Break the loop when user hits the 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
