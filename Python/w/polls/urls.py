from django.urls import path

from . import views

urlpatterns = [
    path('game/pair/', views.game_pairing, name='pair'),
    path('game/<str:id_game>/', views.game, name='game'),
    path('message/', views.message, name='message'),
]