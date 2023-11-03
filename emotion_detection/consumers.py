import cv2
from keras.models import model_from_json
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import threading
import json
import base64
import asyncio

class EmotionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.cap = cv2.VideoCapture(0)
        json_file = open("modal/facialemotionmodel.json", "r")
        self.model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(self.model_json)
        self.model.load_weights("modal/facialemotionmodel.h5")

    async def disconnect(self, close_code):
        self.cap.release()

    async def receive(self, text_data):
        # print(text_data)
        if text_data == "start":
            self.is_processing = True
            await self.process_video()
        
        elif text_data == "stop":
            # print("stop")   
            # await self.is_processing = False
            await self.stop_video_processing()
    
    async def stop_video_processing(self):
        self.is_processing = False
        # print("stop")



  
    async def process_video(self):
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        # await self.send(text_data="Hello world!")
        while self.is_processing:
            success, frame = self.cap.read()
            if not success:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(frame, 1.3, 5)
            try:
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (48, 48))
                    face = face.reshape(1, 48, 48, 1) / 255.0
                    prediction = self.model.predict(face)
                    emotion = labels[np.argmax(prediction)]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()  # Convert to bytes
                await self.send(bytes_data=frame_bytes)
                # await self.send(text_data="Hello world!")
            except Exception as e:
                print("Error in processing video:", e)
            # await self.send(text_data="Hello world2!")
            await asyncio.sleep(0.1)

    async def send_binary(self, event):
        await self.send(event['bytes'], binary=True)
