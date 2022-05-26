import numpy as np # linear algebra
import random

class TicTacToe:
    def __init__(self, player1, player2):
        self.table = np.zeros((3, 3))
        self.turn =  random.randint(-1,1)
        self.player1 = player1
        self.player2 = player2
    def getTurn(self):
        if self.turn == 1:
            return self.player1
        return self.player2
    def changeTurn(self):
        self.turn = self.turn * -1
    def getOwner(self, row, col):
        return self.table.item(row, col)
    def hasOwner(self, row, col):
        return self.table.item(row, col) != 0
    def crossOutCell(self, row, col, player):
        if not self.hasOwner(row, col):
            if self.getTurn() == player:
                self.table.itemset((row, col), self.turn)
                self.changeTurn()
        return False
    def getTable(self):
        print("1:" + self.player1.name)
        print("-1:" + self.player2.name)
        for f in range(0, table.shape[0]):
            for c in range(0, table.shape[1]):
                print(table.item(f, c), end=" ")
        print("\n")
    def someOneWins(self):
        # check rows
        # check diagonals
    def startToPlay(self):
        while True:
            print(self.tictactoe.getTable())
            if self.tictactoe.getTurn().name == self.player1.name:
                self.tictactoe.crossOutCell(1,1, self.player1)
            else:
                self.tictactoe.crossOutCell(1,1, self.player2)
            print(tictactoe.getTable())
class Player:
    def __init__(self, name, isAnAI):
        self.name = name

player1 = Player("Me", isAnAI=False)
player2 = Player("Mark", isAnAI=True)
tictactoe = TicTacToe(player1, player2)
