from tkinter import EW
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import os

def home(request):
  return render(request, os.path.join('ai_core','home.html'), { "user": request.user})