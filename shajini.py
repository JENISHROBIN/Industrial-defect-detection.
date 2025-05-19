import cv2
import matplotlib.pyplot as plt

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load an RGB image (group photo or any image with multiple faces)
image_bgr = cv2.imread("face.jpg")  # Replace with your image path

# Convert BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert to grayscale for detection
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Detect faces: returns list of rectangles (x, y, width, height)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw bounding boxes around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result using matplotlib (so RGB colors are preserved)
plt.imshow(image_rgb)
plt.title(f"Detected Faces: {len(faces)}")
plt.axis('off')
plt.show()