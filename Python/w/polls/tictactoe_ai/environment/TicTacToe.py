from gym.envs.registration import register

# math packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
import math

# openai packages
from gym import Env
from gym.spaces import Discrete, Tuple

# Neural network packages
import tensorflow as tf
from tensorflow import keras
from keras import models, layers

import os
import sys
from numpy._typing import NDArray

# 1. I create the Tic tac toe class and the base configuration of the player
class TicTacToe(Env):
    metadata = {
        "render_modes": [
            "human"
        ],
        "render_fps": 4
    }
    secondTransform = np.flip(np.identity(3), axis=1)
    
    def __init__(self, numOfPlayers: int, table: NDArray, lengthOfLine: int, allowDebug: bool):
        # self.pointUpdated = np.zeros(2) # 0: field, 1 col
        # self.player = 0
        
        self.fields = table.shape[0]
        self.cols = table.shape[1]
        
        # define personalized environment gym props
        self.table = table # has a state role 
        
        #self.action_space = Tuple((Discrete(self.fields), Discrete(self.cols))) # for delete tuples used after
        self.action_space = Discrete(self.fields * self.cols) # for delete tuples used after
        
        # self.action_space = Box(low=-1.0, high=2.0, shape=(self.fields, self.cols), dtype=np.float32)
        #self.action_space.sample()
        
        # self.reset(fillTable=False)
        
        self.lengthOfLine = lengthOfLine
        self.numOfPlayers = numOfPlayers
        self.redirectToChange(allowDebug)
        
    def redirectToChange(self, allowDebug):
        self.redirectPrintTo = sys.stdout

        if not allowDebug:
            self.redirectPrintTo = open(os.devnull, 'w')
    
    def add_players(self, players):
        self.numOfPlayers = len(players)
        self.players = players

    def _get_obs(self):
        return self.table.reshape((1, 9)).tolist()[0]
        # return tf.convert_to_tensor(
        #  [self.table.reshape((1,9)).tolist()],
        #  dtype=tf.int32
        #)
        
    def _get_info(self):
        return {
            "status": None
        }
    
    def reset(self):
        self.table.fill(0)
        self.time = 0
        self.winner = 0
        self.turn = random.choices([1, self.numOfPlayers])[0]

        self.history = tf.Variable(tf.zeros(shape=(self.fields * self.cols + 1, self.fields, self.cols), dtype=tf.dtypes.int32))
        self.saveToHistory()
      
        #return tf.convert_to_tensor(
        #  [self.table.reshape((1,9)).tolist()],
        #  dtype=tf.int32
        #)
        return self.table.reshape((1, 9)).tolist()[0]

    def __deepcopy__(self, memodict={}):
        cpyobj = type(self)() # type: ignore # shallow copy of whole object 
        cpyobj.deep_cp_attr = copy.deepcopy(self.other_attr, memodict) # type: ignore # deepcopy required attr

        return cpyobj
    
    def saveToHistory(self):
        self.history[self.time].assign(self.table)  # type: ignore
        self.time += 1 
            
    def changeTurn(self):
        if self.turn < self.numOfPlayers:
            self.turn = self.turn + 1
        else:
            self.turn = 1

    def hasOwner(self, field: int, col: int):
        return self.table.item(field, col) != 0
 

    def crossOutCell(self, field: int, col: int, idPlayer: int) -> bool:        
        if field < self.table.shape[0] and \
            col < self.table.shape[1] and 0 <= field and 0 <= col:
            if not self.hasOwner(field, col):
                if self.turn == idPlayer:
                    self.table.itemset((field, col), self.turn)
                    self.saveToHistory()
                    self.changeTurn()
        else:
              print("invalid input", file=self.redirectPrintTo)
        return False

    def headerTable(self, players):
        print(str(self.turn) + ", wait for checking to play", file=self.redirectPrintTo)
        if players != None:
            for player in players:
                print(str(player.idPlayer)+":" + player.name, file=self.redirectPrintTo)
        print("", file=self.redirectPrintTo)

    def formatTable(self, players, showHeader: bool = False):
        if showHeader:
              self.headerTable(players)
        for f in range(0, self.table.shape[0]):
            for c in range(0, self.table.shape[1]):
                print(self.table.item(f, c), end=" ", file=self.redirectPrintTo)
            print("", file=self.redirectPrintTo)

    def finishedGame(self) -> bool:
        for f in range(0, self.table.shape[0]):
            for c in range(0, self.table.shape[1]):
                if self.table.item(f, c) == 0:
                    return False
        self.winner = -1
        return True

    def findWinnerAtDiag(self, table: np.matrix) -> int:
        for f in range(0, table.shape[0]):
          #  -table.shape[0]+1 is the last principal diagonal of field
          #  table.shape[1]-1 is the last principal diagonal of cols
          # self.lengthOfLine 
          for numDiag in range(-table.shape[0]+1, table.shape[1]-1):
            diag = np.diag(table, k=numDiag)
            if self.lengthOfLine <= diag.shape[0]:
                player = self.findWinnerAtfields(np.matrix(diag))
                if player != 0:
                    return player
        return 0
  
    def findWinnerAtfields(self, table: np.matrix, redirect=sys.stdout) -> int:
        countOfLine=0
        player=0
        weHaveAWinner=False

        # self.formatTable(table)
        for f in range(0, table.shape[0]):
            player=0
            countOfLine=0

            for c in range(0, table.shape[1]):
                cell = table.item(f, c)
                # print(cell, end=" ")
                if cell == 0:
                    countOfLine=0
                    player=0
                else:
                    if player == cell:
                        countOfLine+=1
                    else:
                        player=cell
                        countOfLine=1
                    if countOfLine == self.lengthOfLine:
                        weHaveAWinner=True
                        break
            # print("")
            if weHaveAWinner:
                break
        if weHaveAWinner:
            return player
        return 0

    def checkWinnerAt(self, findWinnerAt, table, foundMessage):
        winnerAt = findWinnerAt(table) # find winner at function passed
        if winnerAt == 0:
            return False
        else:
            self.winner = winnerAt
            print(foundMessage+ " " + str(winnerAt), file=self.redirectPrintTo)
            return True
    
    def checkIfWinner(self) -> bool:
        print("...checking...", file=self.redirectPrintTo)
        
        # check cols and fields
        return self.checkWinnerAt(self.findWinnerAtfields, self.table, "we have a fields winner!!!,") \
            or self.checkWinnerAt(self.findWinnerAtfields, self.table.transpose(), "we have a cols winner!!!,") \
            or self.checkWinnerAt(self.findWinnerAtDiag, self.table, "we have a principal diagonal winner!!!,") \
            or self.checkWinnerAt(self.findWinnerAtDiag, self.table.dot(TicTacToe.secondTransform), "we have a secondary diagonal winner!!!,")
    
    def scalar_to_action(self, scalar):
        maxField = self.fields
        maxCols = self.cols
        
        field = math.floor(scalar/maxField)
        cols = scalar % maxCols
        
        return (field, cols)
    
    # def step(self, action, idPlayer: int, players):
    def step(self, action):
        # check if you have new points to use
        self.finishedGame()
        self.checkIfWinner()

        if self.winner == 0:
            if type(action) == tuple:
                self.crossOutCell(action[0], action[1], self.turn)  # type: ignore
            elif type(action) == int or type(action) == np.int64:
                new_action = self.scalar_to_action(action)
                self.crossOutCell(new_action[0], new_action[1], self.turn)
            else:
                print("action type error")
                print(type(action))

            self.finishedGame()
            self.checkIfWinner()
            
        return self.getReward(self.turn)
    
      
    def getReward(self, idPlayer):
        reward = 0
        done = False
        info  = {
          "status": "Playing...",
          "winner": 0
        }
                  
        if self.winner != 0:
            if self.winner == -1:
                reward = 0
                done=True
                info = {
                  "status": "We dont have a winner, draw :(",
                  "winner": self.winner
                }

            elif self.winner == idPlayer:
                reward = 1
                done=True
                info = {
                  "status": "You win!",
                  "winner": self.winner
                }
            elif self.winner != idPlayer:
                reward = -1
                done=True
                info = {
                  "status": "You lose",
                  "winner": self.winner
                }
            else:
                print("unknown winner option")

            
        return self.table.reshape((1, 9)).tolist()[0], reward, done, info
    
    def render(self, players=None, mode=None):
        if mode == "human":
            self.formatTable(players, showHeader=True) 
        
    def run_episode(self): # before def invoke
        done = False
        counter = 0
        while not done and counter < 100:
            
            # - 1 because the user of turn 0 doesnt exist
            player = self.players[self.turn - 1]
            
            # render game 
            self.render(self.players, mode="human")
 
            player.play(self.players, tictactoe=self)
            n_state, reward, done, info = self.getReward(player.idPlayer)
       
            counter+=1
        
        self.render(self.players, mode="human")

        # tells all users who wins
        for player in self.players:
            n_state, reward, done, info = self.getReward(player.idPlayer)
            player.save(n_state, reward, done, info)
            
            player.showDescription(self.redirectPrintTo)
            
#register(
#    id='gym_examples/tictactoe',
#    entry_point='gym_examples.envs:tictactoe',
#    max_episode_steps=300,
#)