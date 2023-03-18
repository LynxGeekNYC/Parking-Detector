import cv2
import numpy as np
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

# Parameters
rect_size = 200
min_contour_area = 8000
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=25)
current_cars = {}
sender_email = 'your_email@gmail.com'
password = 'your_password'
receiver_email = 'recipient_email@gmail.com'

# Connect to the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Create email message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = 'Parking Detected!'

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Perform background subtraction to get foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Threshold the foreground mask
    thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and find cars
    for i, contour in enumerate(contours):
        # Ignore small contours
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Check if car is new or existing
        car_found = False
        for j, car in current_cars.items():
            if abs(x - car['x']) < 50 and abs(y - car['y']) < 50:
                car_found = True
                car['countdown'] = 10
                break
        if not car_found:
            # Add new car to current_cars dictionary
            current_cars[i] = {'x': x, 'y': y, 'countdown': 10}

            # Detect color of car in front or behind empty space
            if len(current_cars) > 1:
                current_cars_sorted = sorted(current_cars.items(), key=lambda x: x[1]['x'])
                for k, (n, car) in enumerate(current_cars_sorted):
                    if k == 0 and x > 500:
                        roi = frame[200:200 + rect_size, x - rect_size:x]
                        bgr_mean = cv2.mean(roi)
                        bgr_mean = np.uint8(bgr_mean[:3])
                        hsv_mean = cv2.cvtColor(np.array([[bgr_mean]]), cv2.COLOR_BGR2HSV)[0][0]
                        front_color = hsv_mean
                    if k == len(current_cars_sorted) - 1 and x < 500:
                        roi = frame[200:200 + rect_size, x + 50:x + 50 + rect_size]
                        bgr_mean = cv2.mean(roi)
                        bgr_mean = np.uint8(bgr_mean[:3])
                    hsv_mean = cv2.cvtColor(np.array([[bgr_mean]]), cv2.COLOR_BGR2HSV)[0][0]
                    back_color = hsv_mean

# Decrease countdown for existing cars and remove any cars that have left
for j, car in current_cars.copy().items():
    car['countdown'] -= 1
    if car['countdown'] <= 0:
        # Send email notification when car is missing
        car_rect = (car['x'], car['y'], 50, 50)
        roi = cv2.cvtColor(frame[car_rect[1]:car_rect[1]+car_rect[3], car_rect[0]:car_rect[0]+car_rect[2]], cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi)
        img_io = io.BytesIO()
        pil_image.save(img_io, 'JPEG')
        img_io.seek(0)
        img_data = img_io.read()
        image = MIMEImage(img_data, name='car.jpg')
        msg.attach(image)
        body = MIMEText(f"Car {j} is missing!")
        msg.attach(body)
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, password)
            smtp.send_message(msg)

        # Remove car from current_cars dictionary
        del current_cars[j]

# Show the frame with bounding boxes and colors
cv2.imshow('Parking Detection', frame)

# Wait for 'q' key to exit
if cv2.waitKey(1) & 0xFF == ord('q'):
  
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
