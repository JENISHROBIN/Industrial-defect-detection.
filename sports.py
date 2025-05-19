import cv2
import numpy as np

# Initialize video capture
video_path = "sportss.mp4"  # Replace with 0 for webcam or video file path
cap = cv2.VideoCapture(video_path)

# Define color range for player tracking (e.g., red jersey - adjust for your case)
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

# To store tracked positions
positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 360))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for the selected color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Minimum area threshold to filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            center = (int(x + w/2), int(y + h/2))
            positions.append(center)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

    # Draw trajectory
    for i in range(1, len(positions)):
        cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()