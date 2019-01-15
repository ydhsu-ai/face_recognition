from imutils.video import FPS
import numpy as np
import imutils
import face_recognition
import cv2 as cv

# Load the model 
net = cv.dnn.readNet('face-detection-adas-0001.xml', 'face-detection-adas-0001.bin') 

# Specify target device 
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# capture image from camera
video_capture = cv.VideoCapture(0)

# Initialize some variables

fps = FPS().start()
    
while True:

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Prepare input blob and perform an inference 
    blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U) 
    net.setInput(blob) 
    out = net.forward()
          
    # Draw detected faces on the frame 
    for detection in out.reshape(-1, 7): 
        confidence = float(detection[2]) 
        xmin = int(detection[3] * frame.shape[1]) 
        ymin = int(detection[4] * frame.shape[0]) 
        xmax = int(detection[5] * frame.shape[1]) 
        ymax = int(detection[6] * frame.shape[0])

        if confidence > 0.5:
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
    # FPS
    fps.update()
    fps.stop()
    fps.elapsed()
    font = cv.FONT_HERSHEY_DUPLEX
    cv.putText(frame, "[INFO] approx. FPS: {:.2f}".format(fps.fps()), (20, 30), font, 0.5, (0, 255, 0), 1)

    # Display the resulting image
    cv.imshow('Face Detection', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()
