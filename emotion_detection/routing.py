from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/emotion_detection/$', consumers.EmotionConsumer.as_asgi()),
]