import cv2
import dlib
import imutils
from imutils import face_utils #To get landmarks of eye
import numpy as np
from scipy.spatial import distance # calculate distance between eye landmarks
from pygame import mixer

# Initialize the mixer module
mixer.init()

# Load the sound file
mixer.music.load("D:\music.wav")

# Define a function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set the threshold for the eye aspect ratio and the number of frames to check
thresh = 0.25
frame_check = 20

# Initialize the dlib facial landmark detector and the face predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("D:\shape_predictor_68_face_landmarks.dat")

# Define the indices of the left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Initialize the frame counter
flag = 0

# Start the main loop
while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Resize the frame for better performance
    frame = imutils.resize(frame, width=450)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    subjects = detect(gray, 0)

    # Loop over the detected faces
    for subject in subjects:
        # Predict the facial landmarks for the current face
        shape = predict(gray, subject)

        # Convert the facial landmarks to a NumPy array
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the eye aspect ratio for the left eye
        leftEAR = eye_aspect_ratio(leftEye)

        # Calculate the eye aspect ratio for the right eye
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Check if the eye aspect ratio is below the threshold
        if ear < thresh:
            flag += 1
        else:
            flag = 0

        # If the eye aspect ratio is below the threshold for a certain number of frames, trigger the alarm
        if flag >= frame_check:
            mixer.music.play()
            cv2.putText(frame, "****alert****", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Eye Tracker", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
