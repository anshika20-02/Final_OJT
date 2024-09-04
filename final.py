
# Importing Libraries
import pickle

import cv2
import mediapipe as mp
import numpy as np


# loading the Pre-trained Model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


# start capturing video from the webcam. 0 refers to the default webcam.
cap = cv2.VideoCapture(0)   


# initializing mediapipe hand modules
mp_hands = mp.solutions.hands               #mediaPipe Hands solution for hand tracking
mp_drawing = mp.solutions.drawing_utils     #drawing the hand landmarks on the image
mp_drawing_styles = mp.solutions.drawing_styles   #drawing styles for landmarks and connections


# Configures the hand detection with a minimum detection confidence of 0.3 and enables static image mode.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# dictionary mapping integers to letters of the alphabet (A-Z)
labels_dict = {0: 'A', 1: 'I love You', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Real-time Video Capture and Processing Loop
while True:

    data_aux = []

    ret, frame = cap.read()     #Capturing and Processing Each Frame

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)          #frame to detect hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # hand landmarks are detected, they are drawn on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Initialize lists to store x and y coordinates of landmarks
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

        # Normalize x and y coordinates
        x_min = min(x_)
        y_min = min(y_)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - x_min)
            data_aux.append(y - y_min)

        # Ensure data_aux has the correct number of features (42 in this case)
        data_aux = data_aux[:42]    #preparing Data for Prediction


        # Making the Prediction 
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        #  Displaying the Predicted Character
        cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    # Displaying the Video Feed
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the Webcam and Closing Windows
cap.release()
cv2.destroyAllWindows()