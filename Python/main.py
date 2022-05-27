import numpy as np
from lib.TicTacToe import TicTacToe
from lib.Player import Player

def use_a_menu():
  name1=""
  name2=""
  use_an_ai1=False
  use_an_ai2=False

  insertNames = input('Do you want to insert names?(y/n default n):')
  users = input('How many real users want to play?(0/1/2 default 1 ):')

  if users == "":
    users = 1
  else:
    users = int(users)

  if users == 0:
    use_an_ai1=True
    use_an_ai2=True
  if users == 1:
    use_an_ai2=True

  if insertNames == "n" or insertNames == "":
    name1="Me"
    name2="Mark"

  player1 = Player(name=name1, isAnAI=use_an_ai1)
  player2 = Player(name=name2, isAnAI=use_an_ai2)

  # tictactoe = TicTacToe(player1, player2, np.zeros((3,3)))
  tictactoe = TicTacToe(player1, player2, np.matrix([
    [0,0,0],
    [0,0,0],
    [0,0,0]
  ], np.int32))  # type: ignore
  tictactoe.startToPlay(3)

def no_menu():

  player1 = Player(name="Me", isAnAI=True)
  player2 = Player(name="Mark", isAnAI=True)

  # tictactoe = TicTacToe(player1, player2, np.zeros((3,3)))
  tictactoe = TicTacToe(player1, player2, np.matrix([
    [0,0,0],
    [0,0,1],
    [0,1,0]
  ], np.int32))  # type: ignore
  tictactoe.startToPlay(3)
  # tictactoe.

no_menu()