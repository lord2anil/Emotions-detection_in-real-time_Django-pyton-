# emotion_detection/views.py
from django.shortcuts import render
import cv2
from keras.models import model_from_json
import numpy as np
import base64
import json
from django.http import JsonResponse
import threading

json_file = open("Face_Emotion_Recognition_Machine_Learning-main/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("Face_Emotion_Recognition_Machine_Learning-main/facialemotionmodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion(request):
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    
    def generate_frames():
        webcam = cv2.VideoCapture(0)
        while True:
            _, frame = webcam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            
            try:
                for (p, q, r, s) in faces:
                    image = gray[q:q + s, p:p + r]
                    cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
                    image = cv2.resize(image, (48, 48))
                    img = extract_features(image)
                    pred = model.predict(img)
                    prediction_label = labels[pred.argmax()]
                    cv2.putText(frame, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
                _, buffer = cv2.imencode('.jpg', frame)
                frame = base64.b64encode(buffer).decode('utf-8')
                yield (f'data:image/jpeg;base64,{frame}')
            except cv2.error:
                pass

    return render(request, 'home.html')

def get_frame(request):
    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    while True:
        _, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        try:
            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(frame, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            _, buffer = cv2.imencode('.jpg', frame)
            frame = base64.b64encode(buffer).decode('utf-8')
            data = {
                'frame': f'data:image/jpeg;base64,{frame}',
                'emotion': prediction_label,
            }
            return JsonResponse(data)
        except cv2.error:
            pass

# Create a thread to continuously update the frame
frame_thread = threading.Thread(target=get_frame)
frame_thread.daemon = True
frame_thread.start()
