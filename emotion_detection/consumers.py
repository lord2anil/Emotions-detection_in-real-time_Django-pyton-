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
        # self.cap = cv2.VideoCapture(0)
        json_file = open("model/facialemotionmodel.json", "r")
        self.model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(self.model_json)
        self.model.load_weights("model/facialemotionmodel.h5")
        self.is_processing = False
        self.chunks=[]

    # async def disconnect(self, close_code):
    #     self.cap.release()

    async def receive(self, text_data=None, bytes_data=None, **kwargs):
          labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
          
          if(text_data):
            
         

            self.chunks.append( text_data)
            # print(len(self.chunks))
            if len(self.chunks) > 0:
                frame_str = self.chunks.pop(0)
                
                  #working 


                frame = cv2.imdecode(np.frombuffer(base64.b64decode(frame_str), np.uint8), cv2.IMREAD_COLOR)

                # Convert the frame to grayscale
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
                # await asyncio.sleep(0.01)
                
            
    
    async def stop_video_processing(self):
        self.is_processing = False
        # print("stop")



  
    async def process_video(self):
        print(len(self.chunks))

        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        # await self.send(text_data="Hello world!")
        if len(self.chunks) > 0:
            # success, frame = self.cap.read()
            frame_base64 = self.chunks.pop(0)
            print(frame_base64)
            frame_bytes = base64.b64decode(frame_base64)

            # if not success:
            #     break
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
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
            await asyncio.sleep(0.01)

    async def send_binary(self, event):
        await self.send(event['bytes'], binary=True)
