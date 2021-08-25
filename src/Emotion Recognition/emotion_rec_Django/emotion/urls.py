"""emotion URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .view import *

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('',Home, name = 'app1fun'),
    path('frame_feed', frame_feed, name='frame_feed'),
    path('text_bar_chart', text_bar_chart, name='text_bar_chart'),
    path('data1', data1, name='data1'),
path('f_bar_chart', f_bar_chart, name='f_bar_chart'),
path('audio_bar_stream', audio_bar_stream, name='audio_bar_stream'),
]
