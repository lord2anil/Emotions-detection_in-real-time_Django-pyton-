"""
ASGI config for face_detection project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

# import os

# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_detection.settings')

# application = get_asgi_application()

# # emotion_detection_project/asgi.py

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import emotion_detection.routing  # Import your WebSocket routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_detection.settings')
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            emotion_detection.routing.websocket_urlpatterns
        )
    ),
})
