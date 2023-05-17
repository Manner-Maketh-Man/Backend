from django.urls import path
from . import views

urlpatterns = [
    path('process_json/', views.process_json, name='process_json'),
]
