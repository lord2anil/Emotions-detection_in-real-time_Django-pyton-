# emotion_detection/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('detect_emotion/', views.detect_emotion, name='detect_emotion'),
    path('get_frame/', views.get_frame, name='get_frame'),
]
