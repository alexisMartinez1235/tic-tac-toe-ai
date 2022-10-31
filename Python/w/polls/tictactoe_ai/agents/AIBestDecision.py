# from tensorflow import keras
# from keras import losses, metrics, optimizers

import sys
from turtle import Turtle
from keras.models import Sequential
from keras.layers import MaxPooling2D, \
    Dropout, \
    Dense, \
    Flatten, \
    Convolution2D as Conv2D, \
    Reshape
from keras.utils.vis_utils import plot_model
from Controller import createPlot, playTwoPlayers
from agents.AI_Reinforcement_Random import AI_Reinforcement_Random
from agents.Player import Player

import tensorflow as tf

# from keras.optimizers import SGD
import pandas as pd
from pandas import DataFrame

import numpy as np

from pathlib import Path
from IPython.display import Image

from matplotlib import pyplot as plt

from environment.TicTacToe import TicTacToe
BASE_DATASETS = ""
# BASE_DATASETS = "/kaggle/" # on remote notebooks

class AIBestDecision(AI_Reinforcement_Random):
    _instance = None
    _initialized = False
    network: Sequential

    def __init__(self, idPlayer, name: str, \
            datasetFile = BASE_DATASETS + 'input/tictactoe-win-movement/tictactoe_win_movement.csv', \
            datasetFileUpdates = BASE_DATASETS + 'working/tictactoe_win_movement.csv', \
                ):
        super().__init__(idPlayer, name)
        
        self.inputTable = []
        self.prob_choose = []
        self.turn_choose  = []
        self.next_table = []

        # self.network = None
        
        self.metrics = None
        self.datasetFile = datasetFile
        self.datasetFileUpdates = datasetFileUpdates

        self.numOfGames = 0
    
    def __new__(cls, *args):
      if cls._initialized:
        network = cls._instance.network # save network
      
      cls._instance = super(AIBestDecision, cls).__new__(cls)
      cls._instance.__init__(*args)

      if cls._initialized:
        cls._instance.network = network # replace network

      if not cls._initialized:
        
        print('Training a new ai best decision instance')

        cls._instance.initial_structure(True)
        
        cls._initialized = True 

      return cls._instance
        
    def initial_structure(self, build):
        
        if build:
            
            # create dataset file
            self.createDataset()

            # create data
            self.collect()
                        
            # save already created data
            self.saveToDataset()
            
            # get already saved data
            self.getDataset()
            
            # create model, compile and fit
            self.model()
            self.compileNetwork()
            self.fit()
            
    def collect(self):
        for idGame in range(0, self.numOfGames):
            # history, rewardsPerUser = self.randomPlay()
            history, winner = self.randomPlay()
            length_of_history = len(history.numpy()) # type: ignore

            for i in range(0, length_of_history - 1):

                tableBefore = np.matrix(history[i])  # type: ignore
                tableAfter = np.matrix(history[i+1]) # type: ignore

                # if tableAfter is not zero, T != 0
                if not np.equal(tableAfter, np.zeros((3,3))).all():

                    tableChange = (tableAfter - tableBefore)

                    # 1/self.idPlayer normalze the matrix ( all 0 or 1)
                    turn = tableChange.sum()
                    tableChangeNormalized = (1/turn) * tableChange

                    # If the winner plays and he is not me, do roles reverse.
                    # that is instead of ignoring 
                    if winner != 1:
                        
                        # learn from the rival decisions
                        tableBefore = self.swap_id_player_in_table(tableBefore, winner, 1)
                        tableAfter= self.swap_id_player_in_table(tableAfter, winner, 1)
                        
                    # if win, draw or lose in the actual table, too in identity table
                    self.inputTable.append(tableBefore)  # type: ignore
                    self.prob_choose.append(tableChangeNormalized) # type: ignore
                    self.turn_choose.append(turn) # type: ignore
                    self.next_table.append(tableAfter)
    
                    # if win, draw or lose in the actual table, too in table transpose
                    self.inputTable.append(tableBefore.copy().transpose())  # type: ignore
                    self.prob_choose.append(tableChangeNormalized.copy().transpose())  # type: ignore
                    self.turn_choose.append(turn)  # type: ignore
                    self.next_table.append(tableAfter.copy().transpose())
                    
                    # if win, draw or lose in the actual table, too in table flip 
                    self.inputTable.append(tableBefore.copy().dot(TicTacToe.secondTransform))  # type: ignore
                    self.prob_choose.append(tableChangeNormalized.copy().dot(TicTacToe.secondTransform))  # type: ignore
                    self.turn_choose.append(turn)  # type: ignore
                    self.next_table.append(tableAfter.copy().dot(TicTacToe.secondTransform))
                    
                    
                    # if win, draw or lose in the actual table, too in transpose and flips
                    
                    self.inputTable.append(  # type: ignore
                        tableBefore.copy()
                            .dot(TicTacToe.secondTransform)
                            .transpose()
                    )
                    self.prob_choose.append(  # type: ignore
                        tableChangeNormalized.copy()
                            .dot(TicTacToe.secondTransform) 
                            .transpose()
                    )
                    self.turn_choose.append(turn) # type: ignore
                    self.next_table.append(
                        tableAfter.copy()
                            .dot(TicTacToe.secondTransform)
                            .transpose()
                    )
                    
                    
    def createDataset(self):
        # /kaggle/working
        # if not Path(self.datasetFileUpdates).is_file():
        csvBestDecision = DataFrame(index=None, columns=["table", "table_prob_choose", "turn_choose", "next_table"] ,dtype=None, copy=True)  
        csvBestDecision.to_csv(self.datasetFileUpdates, index=False) # save to notebook output
            
    def getDataset(self):
        csvBestDecision = pd.read_csv(self.datasetFile) # load from notebook input
        
        self.inputTable = csvBestDecision["table"].to_list()
        self.prob_choose =  csvBestDecision["table_prob_choose"].to_list()
        self.turn_choose =  csvBestDecision["turn_choose"].to_list()
        
        for i in range(0, len(self.inputTable)):
            # tableAux = np.matrix(AIBestDecision.inputTable[i], dtype= np.int32)
            # prob_choose_aux = np.matrix(AIBestDecision.prob_choose[i], dtype=np.int32)

            self.inputTable[i] = np.reshape(
                np.matrix(self.inputTable[i], dtype= np.int32),
                (3, 3)
            )
            self.prob_choose[i] = np.reshape(
                np.matrix(self.prob_choose[i], dtype=np.int32),
                (1, 9)
            )
            
        self.inputTable = np.array(self.inputTable)
        self.prob_choose =  np.array(self.prob_choose)
        self.turn_choose =  np.array(self.turn_choose)

        # AIBestDecision.inputTable = tf.convert_to_tensor(AIBestDecision.inputTable)
        # AIBestDecision.prob_choose =  tf.convert_to_tensor(AIBestDecision.prob_choose)
        
        print("Data size: " + str(len(csvBestDecision["table_prob_choose"])))
        
    def saveToDataset(self):
        data = {
            'table': self.inputTable,
            'table_prob_choose': self.prob_choose,
            "turn_choose": self.turn_choose,
            "next_table": self.next_table
        }
                
        # Make data frame of above data
        df = pd.DataFrame(data)
 
        # append data frame to CSV file
        df.to_csv(self.datasetFileUpdates, mode='a', header=False, index=False)
        
    def play(self, players, tictactoe):
        table = tictactoe.table.copy()
        action = (0, 0)
        probToWin = 0
        
        # adapt the table to the train data
        if self.idPlayer != 1:
            table = self.swap_id_player_in_table(table, self.idPlayer, 1)
            
        predict = self.network.predict(np.array(table).reshape(
            (1, tictactoe.fields, tictactoe.cols)
        )).reshape((tictactoe.fields, tictactoe.cols))
        
        for f in range(0, tictactoe.fields):
            for c in range(0, tictactoe.cols):
                
                # search empty spaces in the table 
                if table.item(f,c) == 0:
                    
                    # search best option
                    if probToWin < predict.item(f, c):
                        probToWin = predict.item(f, c)
                        action = (f, c)
                        
        n_state, reward, done, info = tictactoe.step(action)
        
        self.save(n_state, reward, done, info)
        
    def randomPlay(self):
        # Player
        tictactoe = TicTacToe(2, np.zeros((3, 3), np.int32), 3, allowDebug=False)  # type: ignore

        player1 = AI_Reinforcement_Random(1, "Train1")
        player2 = AI_Reinforcement_Random(2, "Train2")


        tictactoe.add_players([player1, player2])
        tictactoe.reset()
        tictactoe.run_episode()

        # return [ tictactoe.history, [ player1.reward, player2.reward ] ]
        return tictactoe.history, tictactoe.winner
    
    def divide_data(self):
        trainBatch = int(2*len(self.prob_choose)/3)
        print(trainBatch)
        
        input_train = self.inputTable[0:trainBatch]
        prob_choose_train = self.prob_choose[0:trainBatch]
     
        input_val = self.inputTable[trainBatch:len(self.prob_choose)]
        prob_choose_val = self.prob_choose[trainBatch:len(self.prob_choose)]
            
        print("train")
        print(len(input_train))
        print(len(prob_choose_train))
        
        print("eval")
        print(len(input_val))
        print(len(prob_choose_val))

        return trainBatch, input_train, prob_choose_train, input_val, prob_choose_val
    
    def model(self):
        
        self.network = Sequential()
        self.network.add(Dense(3, input_shape=(3, 3), activation='relu'))
        
        self.network.add(Dense(6, activation='relu')) # ...
        self.network.add(Dense(9, activation='relu')) # 30 
        self.network.add(Dense(30, activation='relu')) # 30  
        self.network.add(Dense(30, activation='relu')) # 30  
        self.network.add(Dense(30, activation='relu'))
        self.network.add(Dense(18, activation='relu'))
        
        
        # AIBestDecision.network.add(Dense(9, activation='sigmoid'))
        self.network.add(Reshape((1, 18*3)))
        self.network.add(Dense(9, activation='softmax'))
        
        self.network.summary()
        
    def compileNetwork(self):

        self.network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # self.network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def fit(self):
        trainBatch, input_train, prob_choose_train, input_val, prob_choose_val = self.divide_data()
        num_epoch = 10 # 35

        self.metrics = self.network.fit(
            np.array(input_train),
            np.array(prob_choose_train),
            epochs=num_epoch,
            batch_size=int(trainBatch/num_epoch), # /num_epoch
             validation_data=(
                np.array(input_val),
                np.array(prob_choose_val)
            ),
            verbose=1  # type: ignore
        )
        
        self.seeMetrics()
        
    def seeMetrics(self):
        
        # Plot the accuracy curves
        plt.plot(self.metrics.history['accuracy'],'bo') # type: ignore
        plt.plot(self.metrics.history['val_accuracy'],'rX') # type: ignore
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid()
        plt.show()
        
        # summarize history for loss
        plt.plot(self.metrics.history['loss'],'bo') # type: ignore
        plt.plot(self.metrics.history['val_loss'],'rX') # type: ignore
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.grid()
        plt.show()

