import numpy as np
import cv2 as cv

# Load the cascade
haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Mike Iannotti', 'Elton John']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# 0 for webcam
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
while True:
    
    # capture each frame
    ret, frame = cap.read()
    # convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect the face in the frame
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for(x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+h]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    
    cv.imshow('Detected Face', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#     # Release the capture and close all windows
cap.release()
cv.destroyAllWindows()




# # Function to detect faces and draw rectangles
# def detect_faces_in_video():
#     # Open webcam
#     cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    
#     while True:
#         # Read the frame
#         ret, frame = cap.read()
        
#         # convert frame to grayscale
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
#         # Detect faces in grayscale fram
#         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
#         # Draw rectangle around each face
#         for (x, y, w, h) in faces:
#             cv.putText(frame, 'Mike Iannotti', (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
#             cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # Display the output with rectangles and text added
#         cv.imshow('Webcam - Press "q" to exit', frame)
        
#         # Exit on 'q' key press
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release the capture and close all windows
#     cap.release()
#     cv.destroyAllWindows()

# # Call the function to detect faces in video
# detect_faces_in_video()