import numpy as np
from lib.TicTacToe import TicTacToe
from lib.Player import Player

# player1 = Player("Me", isAnAI=False)
player1 = Player("Jessica", isAnAI=True)
player2 = Player("Mark", isAnAI=True)

# tictactoe = TicTacToe(player1, player2, np.zeros((3,3)))
tictactoe = TicTacToe(player1, player2, np.matrix([
  [0,0,0],
  [0,0,1],
  [0,1,0]
], np.int32))  # type: ignore
tictactoe.startToPlay(3)
# tictactoe.