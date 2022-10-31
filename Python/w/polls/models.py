# import datetime

from django.db import models
from django.contrib.auth.models import User

# from django.utils import timezone

# Create your models here.

class Player(models.Model):
  username = models.OneToOneField(User, to_field="username", on_delete=models.CASCADE, unique=True, null=True)
  is_robot = models.BooleanField(default=False)

  def __str__(self):
    return "player has username %s %s" % (self.username, super().__str__())

class PairTicTacToe(models.Model):
    room_code = models.CharField(max_length=100, unique=True, null=True)
    game_creator = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="game_creator", blank=False, null=True)
    game_opponent = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="game_opponent", blank=False, null=True)
    is_over = models.BooleanField(default=False)

    def __str__(self):
      res = "The %s room has tha players %s with the opponnet %s" % (self.room_code, self.game_creator, self.game_opponent)
      
      if self.is_over:
        res += "and the game is over"
      else:
        res += "and the game is waiting for finish"

      return res 