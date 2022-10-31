import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# from polls.forms import PairTicTacToeForm
from allauth.account.decorators import verified_email_required
from django.views.generic import CreateView

from polls.forms import PairGameForm

def game(request):
    if request.method == 'POST':
        form = PairGameForm(request.POST)
        if form.is_valid():
            return render(request, '/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = PairGameForm()

    return render(request, 'polls/game.html', {'form': form})

# Create your views here.
def game_pairing(request):
    if request.method == 'POST':
      form = PairGameForm(request.POST)
      if form.is_valid():
        pass
    else:
      form = PairGameForm()
    return render(request, 'polls/forms.html', {'form': form })

@verified_email_required
def message(request):
  return JsonResponse({
    "message": "Hello world!"
  })

def questions(request):
  pass