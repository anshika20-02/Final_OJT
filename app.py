from flask import Flask, render_template, Response, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# Loading the Pre-trained Model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {i: chr(65 + i) for i in range(26)}

# Initialize global variables
predicted_letters = []
current_word = []
letter_to_add = None
frame_count = 0
frames_for_sign = 15
last_sign_time = time.time()
sign_timeout = 2
confirmed_letter = None
predicted_character = ' '

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
    clear_sentence()
    clear_word()

def generate_frames():
    global predicted_character, confirmed_letter, letter_to_add, frame_count, last_sign_time

    cap = cv2.VideoCapture(0)
    
    while True:
        data_aux = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

            x_min = min(x_)
            y_min = min(y_)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - x_min)
                data_aux.append(y - y_min)

            data_aux = data_aux[:42]

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            if letter_to_add == predicted_character:
                frame_count += 1
            else:
                letter_to_add = predicted_character
                frame_count = 1

            if frame_count >= frames_for_sign:
                confirmed_letter = predicted_character
                letter_to_add = None
                frame_count = 0
                last_sign_time = time.time()

                if confirmed_letter == '<CLR>':
                    clear_all()
                elif confirmed_letter == ' ':
                    add_word_to_sentence()
                    clear_word()
                else:
                    add_letter_to_word(confirmed_letter)
                confirmed_letter = None

        else:
            if time.time() - last_sign_time > sign_timeout:
                add_word_to_sentence()
                clear_word()

            predicted_character = ' '

        sentence = get_sentence()
        word = get_word()
        cv2.putText(frame, f"Predicted Letter: {predicted_character}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Word: {word}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Sentence: {get_sentence()}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence_route():
    clear_sentence()
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
