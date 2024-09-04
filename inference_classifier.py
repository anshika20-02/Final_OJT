# Importing Libraries
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Loading the Pre-trained Model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start capturing video from the webcam. 0 refers to the default webcam.
cap = cv2.VideoCapture(0)   

# Initializing mediapipe hand modules
mp_hands = mp.solutions.hands               # MediaPipe Hands solution for hand tracking
mp_drawing = mp.solutions.drawing_utils     # Drawing the hand landmarks on the image
mp_drawing_styles = mp.solutions.drawing_styles   # Drawing styles for landmarks and connections

# Configures the hand detection with a minimum detection confidence of 0.3 and enables static image mode.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary mapping integers to letters of the alphabet (A-Z)
labels_dict = {i: chr(65 + i) for i in range(26)}

# Initialize global variables
predicted_letters = []
current_word = []
letter_to_add = None
frame_count = 0
frames_for_sign = 15  # Number of frames to confirm a consistent sign
last_sign_time = time.time()
sign_timeout = 2  # Time in seconds to determine if a new word should start
confirmed_letter = None
predicted_character = ' '  # Initialize with a default value

def add_letter_to_word(letter):
    global current_word
    current_word.append(letter)

def add_word_to_sentence():
    global predicted_letters, current_word
    if current_word:
        predicted_letters.append(''.join(current_word))
        current_word = []

def clear_word():
    global current_word
    current_word = []

def clear_sentence():
    global predicted_letters
    predicted_letters = []
    global current_word
    current_word = []

def get_word():
    global current_word
    return ''.join(current_word)

def get_sentence():
    global predicted_letters
    return ' '.join(predicted_letters)

def clear_all():
    clear_sentence()  # Clear the entire sentence
    clear_word()      # Clear the current word

# Real-time Video Capture and Processing Loop
while True:
    data_aux = []

    ret, frame = cap.read()  # Capturing and Processing Each Frame

    if not ret:
        print("Failed to capture image from the webcam.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)  # Frame to detect hand landmarks
    if results.multi_hand_landmarks:
        # Hand landmarks are detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Hand landmarks are drawn on the frame
            mp_drawing.draw_landmarks(
                frame,  # Image to draw
                hand_landmarks,  # Model output
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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
        data_aux = data_aux[:42]  # Preparing Data for Prediction

        # Making the Prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Check if the predicted character is the same for a number of frames
        if letter_to_add == predicted_character:
            frame_count += 1
        else:
            letter_to_add = predicted_character
            frame_count = 1

        if frame_count >= frames_for_sign:
            # Confirm the letter and add it to the word
            confirmed_letter = predicted_character
            # Reset frame count and letter_to_add
            letter_to_add = None
            frame_count = 0
            last_sign_time = time.time()

            if confirmed_letter == '<CLR>':
                clear_all()  # Clear the entire sentence and word
            elif confirmed_letter == ' ':
                add_word_to_sentence()
                clear_word()
            else:
                add_letter_to_word(confirmed_letter)
            # Reset confirmed_letter
            confirmed_letter = None

    else:
        # No hand detected, check if the timeout for adding a new word has been reached
        if time.time() - last_sign_time > sign_timeout:
            add_word_to_sentence()
            clear_word()

        # Reset the predicted_character if no hand landmarks are detected
        predicted_character = ' '
        
    # Display the current status on the frame
    sentence = get_sentence()
    word = get_word()
    cv2.putText(frame, f"Predicted Letter: {predicted_character}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Word: {word}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Sentence: {get_sentence()}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    # Displaying the Video Feed
    cv2.imshow('frame', frame)
    
    # Check for key press events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Press 'c' to clear the sentence
        clear_all()

# Releasing the Webcam and Closing Windows
cap.release()
cv2.destroyAllWindows()
