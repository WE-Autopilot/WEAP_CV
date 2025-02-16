import cv2
import numpy as np

# Read the test video, currently testing
cap = cv2.VideoCapture('datasets/test/F1TenthOnboardVid.mp4')

# Create the orb object for ORB feature detection and extraction
# orb (oriented fast and rotation brief) is a feature extractor and matching algorithm
# it's lightweight and efficient. Provides 3D-esque representation
orb = cv2.ORB_create(1000) 

# main loop
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Checking the dimensions of the video, they are 720 x 1280 x 3
    '''
    h, w, c = frame.shape
    print(f"Height: {h} Width: {w} channels: {c} ")
    '''
    
    # Pre-processing steps, need to add more for lens distortion 
    ### ****************** ###
    frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                       interpolation=cv2.INTER_CUBIC)
    # Convert the frame to gray so that ORB and Canny edge detection can work
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    ### ****************** ###


    # Compute the orb detections
    keypoints, descriptors = orb.detectAndCompute(grayFrame, None) 
    # The actual frame with the detections drawn on top
    frame_with_kp = cv2.drawKeypoints(grayFrame, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imshow("ORB Keypoints", frame_with_kp)

    # Canny edge detection, black and white representation of edges in a frame
    canny = cv2.Canny(grayFrame, 100, 200)
    cv2.imshow("Canny edge detection", canny)

    # Display the resulting frame
    #cv2.imshow('F1tenth Onboard Video', frame)

     # define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break   

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()