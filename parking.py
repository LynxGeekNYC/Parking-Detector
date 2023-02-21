import cv2
import numpy as np
import smtplib, ssl

# Define the dimensions of the rectangle
rect_width = 100
rect_height = 100

# Define the lower and upper bounds for the white color
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 20, 255])

# Define the email parameters
smtp_server = "smtp.gmail.com"
port = 587
sender_email = "your_email@gmail.com"
receiver_email = "recipient_email@gmail.com"
password = "your_email_password"
message = """\
Subject: Parking Available

Parking Spot is available at """

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the state dictionary for each rectangle
rect_state = {}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the image to extract only the white pixels
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find the contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and label each rectangle
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == rect_width and h == rect_height:
            # Check if the rectangle is completely covered by a different color
            rect_pixels = mask[y:y+h, x:x+w]
            rect_area = w * h
            rect_filled = cv2.countNonZero(rect_pixels) / rect_area
            rect_id = f"{x}-{y}-{w}-{h}"
            if rect_id in rect_state:
                # Check if the rectangle has changed color or filled up most of the way
                if rect_state[rect_id]["filled"] != rect_filled or rect_state[rect_id]["color"] != "white":
                    # Send an email notification
                    context = ssl.create_default_context()
                    with smtplib.SMTP(smtp_server, port) as server:
                        server.starttls(context=context)
                        server.login(sender_email, password)
                        server.sendmail(sender_email, receiver_email, message)
            rect_state[rect_id] = {"filled": rect_filled, "color": "white"}
            # Draw a labeled rectangle around the detected rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, rect_id, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
