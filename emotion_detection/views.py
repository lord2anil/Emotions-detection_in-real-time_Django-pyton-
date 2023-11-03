# emotion_detection/views.py
from django.shortcuts import render
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.http import JsonResponse
import cv2  # Import OpenCV here
import base64
import numpy as np  # Import NumPy here
from keras.models import model_from_json

# Load the emotion detection model
json_file = open("modal/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("modal/facialemotionmodel.h5")

# Create an instance of the CascadeClassifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@async_to_sync
async def process_video_consumer(message):
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        try:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face.reshape(1, 48, 48, 1) / 255.0
                prediction = model.predict(face)
                emotion = labels[np.argmax(prediction)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')

            # Send the frame to the client via the WebSocket
            await message.reply_channel_layer.group_send(
                message.reply_channel,
                {
                    "type": "send_frame",
                    "frame": frame_bytes,
                },
            )
        except Exception as e:
            print(e)

def detect_emotion(request):
    return render(request, 'home.html')

def start_emotion_detection(request):
    # Start the video processing in the background
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.send)({
        "type": "process_video_consumer",
    })
    return JsonResponse({"message": "Emotion detection started"})

def stop_emotion_detection(request):
    # You can add logic to stop the video processing here if needed
    return JsonResponse({"message": "Emotion detection stopped"})
