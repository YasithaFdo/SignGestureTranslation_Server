from flask import Flask, jsonify , request
import base64
import cv2
import numpy as np
import os
import joblib
from flask_cors import CORS
from cvzone.HandTrackingModule import HandDetector
from sympy import false
import math
import mediapipe as mp
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)
CORS(app)

# Hand Detector
detector = HandDetector(maxHands=1)
offset = 30
imgSize = 300

categories = ['apple', 'big', 'call', 'can', 'drink', 'eat', 'good', 'have', 'headache', 'house', 'joke', 'laugh', 'look', 'love', 'me', 'pee', 'please', 'pull', 'shut up', 'sleep', 'small', 'sorry', 'stop', 'tree', 'walk', 'welcome', 'you']

# Constants
HAND_FEATURES_LENGTH = 21 * 3 * 2
FACE_FEATURES_LENGTH = 4 + (5 * 2)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# BART model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

 #Load models
save_dir = "Models"
pathSVM = os.path.join(save_dir, "SVMclassifierF&H_Raw.pkl")

svm_model = joblib.load(pathSVM)

@app.route("/")
def index():
    return "Hello World"

@app.route('/getting', methods=['GET'])
def getting():
    return jsonify({"status": "Success", "svm_prediction": "Hiiiii"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_base64 = data.get('image')

        imgSplit = image_base64.split(",")[1]  # Remove 'data:image/png;base64,' part

        # Fix Base64 Padding Issue
        imgSplit += "=" * ((4 - len(imgSplit) % 4) % 4)  # Ensures correct padding

        # Decode Base64
        image_data = base64.b64decode(imgSplit)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        hands, img = detector.findHands(img)

        # Detect faces using MediaPipe Face Mesh
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(img_rgb)

        # Initialize feature vector
        all_landmarks = []

        if hands:
            for hand in hands[:2]:
                lmList = hand['lmList']
                flat_lmList = np.array(lmList).flatten()
                max_value = np.max(flat_lmList)
                if max_value > 0:
                    flat_lmList = flat_lmList / max_value
                all_landmarks.extend(flat_lmList)

            while len(all_landmarks) < HAND_FEATURES_LENGTH:
                all_landmarks.extend([0] * (21 * 3))

            # Process face landmarks
            face_features = [0] * FACE_FEATURES_LENGTH
            face_near_hand = 0

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Get bounding box
                    bboxC = face_landmarks.landmark
                    x_min = min([lm.x for lm in bboxC]) * img.shape[1]
                    y_min = min([lm.y for lm in bboxC]) * img.shape[0]
                    x_max = max([lm.x for lm in bboxC]) * img.shape[1]
                    y_max = max([lm.y for lm in bboxC]) * img.shape[0]
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Store bounding box
                    face_features[:4] = [x_min / img.shape[1], y_min / img.shape[0], bbox_width / img.shape[1],
                                         bbox_height / img.shape[0]]

                    # Extract key facial landmarks (eyes, nose, mouth)
                    key_landmark_indices = [33, 263, 1, 61, 291]  # Right eye, Left eye, Nose tip, Mouth left, Mouth right
                    for i, lm_idx in enumerate(key_landmark_indices):
                        face_features[4 + (i * 2)] = bboxC[lm_idx].x
                        face_features[4 + (i * 2) + 1] = bboxC[lm_idx].y

                    # Check if hand is near the face
                    for hand in hands:
                        hand_center_x = (hand['bbox'][0] + hand['bbox'][2]) // 2
                        hand_center_y = (hand['bbox'][1] + hand['bbox'][3]) // 2
                        face_center_x = (x_min + x_max) / 2
                        face_center_y = (y_min + y_max) / 2
                        distance = math.sqrt((hand_center_x - face_center_x) ** 2 + (hand_center_y - face_center_y) ** 2)
                        if distance < 100:
                            face_near_hand = 1

            # Concatenate features
            all_landmarks.extend(face_features)
            all_landmarks.append(face_near_hand)

            # Ensure feature vector has consistent length
            expected_length = HAND_FEATURES_LENGTH + FACE_FEATURES_LENGTH + 1
            if len(all_landmarks) == expected_length:
                # Reshape for prediction
                all_features = np.array(all_landmarks).reshape(1, -1)

                # Predict using SVM
                svm_prediction = svm_model.predict(all_features)[0]
                label = categories[svm_prediction]
                print(label)
                # Display predictions on the image
                # cv2.putText(img, f"SVM: {svm_label}", (10, 50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                return jsonify(
                    {"status": "Success", "svm_prediction": label})
        else:
            return jsonify({"status":"No Hands"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/textgenerate', methods=['POST'])
def textgenerate():
    try:
        data = request.get_json()
        predicted_words = data.get('predicted_words')  # Get predicted words from the request

        if not predicted_words:
            return jsonify({"status": "error", "message": "No predicted words provided"})



        # Join the predicted words to form a sentence with <mask> in between
        masked_sentence = " <mask> ".join(predicted_words)

        # Tokenize the masked sentence
        tokenized_sent = tokenizer(masked_sentence, return_tensors='pt')

        # Generate refined output using BART
        generated_encoded = bart_model.generate(
            tokenized_sent['input_ids'],
            max_length=50,  # Allow room for a full sentence
            num_beams=5,  # Use beam search for better predictions
            early_stopping=True
        )

        # Decode the generated sentence
        refined_sentence = tokenizer.decode(generated_encoded[0], skip_special_tokens=True)
        print(refined_sentence)

        # Return the generated sentence
        return jsonify({"status": "Success", "generated_sentence": refined_sentence})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
