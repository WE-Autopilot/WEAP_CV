import cv2
import numpy as np
from ultralytics import YOLO

# initialize the detection model, classification add: -cls, segementiaton add: -seg
model = YOLO("yolov8n.pt")

# Read the test video
cap = cv2.VideoCapture('datasets/test/F1TenthOnboardVid.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                       interpolation=cv2.INTER_CUBIC)

    # Display the resulting frame, commenting out for performance purposes
    # cv2.imshow('Frame', frame)

    # Run the model on each frame
    results = model(frame)

     # Draw detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls_id = int(box.cls[0])  # Class ID
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()