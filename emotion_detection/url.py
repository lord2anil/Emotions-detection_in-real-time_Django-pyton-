
from django.urls import path
from . import views

urlpatterns = [
    path('detect_emotion/', views.detect_emotion, name='detect_emotion'),
]