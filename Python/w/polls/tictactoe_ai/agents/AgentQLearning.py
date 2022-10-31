import sys
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf

from tensorflow import keras
from keras import layers

from keras import losses, metrics, optimizers
from Controller import createPlot, playTwoPlayers
from agents.AI_Reinforcement_Random import AI_Reinforcement_Random
from environment.TicTacToe import TicTacToe

# from keras.optimizers import Adam  
from keras.models import Sequential
from keras.layers import (
  MaxPooling2D,
  Dropout,
  Dense,
  Flatten,
  InputLayer,
  Convolution2D as Conv2D,
  Reshape
)
from keras.utils.vis_utils import plot_model
# from tf_agents.agents import DdpgAgent

import numpy as np

# class AgentQLearning(AIBestDecision):
class AgentQLearning(AI_Reinforcement_Random):
  _instance = None
  _initialized = False
  network: Sequential
  agent: DQNAgent

  def __init__(self, idPlayer, name, tictactoe):
    super().__init__(idPlayer, name)
    self.tictactoe = tictactoe
    self.states = self.tictactoe.fields * self.tictactoe.cols
    self.actions = (self.tictactoe.action_space).n
    #self.actions =  [
    #  self.tictactoe.action_space[0].n,
    #  self.tictactoe.action_space[1].n
    #]
    
    self.policy = None
    self.memory = None
    # self.agent= None
    self.model = None
    
    # if self.policy == None:
    # self.build_agent()

  def __new__(cls, *args):
    if cls._initialized:
      network = cls._instance.network # save network
      agent = cls._instance.agent # save agent
    
    cls._instance = super(AgentQLearning, cls).__new__(cls)
    cls._instance.__init__(*args)


    if cls._initialized:
      cls._instance.network = network # restore network
      cls._instance.agent = agent # restore agent

    if not cls._initialized:
      
      print('Training a new ai best decision instance')

      cls._instance.build_model()
      cls._instance.train()
      cls._instance.test()
      
      cls._initialized = True 

    return cls._instance


  def build_model(self):
    self.model = Sequential()
    
    #self.model.add(Dense(2, input_shape=(1, 2), activation='relu'))
    #self.model.add(Flatten(input_shape=(1, 9)))
    
    self.model.add(InputLayer(input_shape=(1, 9)))
    self.model.add(Flatten())
    
    #self.model.add(Dense(10, activation='relu'))
    #self.model.add(Dense(10, activation='relu'))
    #self.model.add(Dense(10, activation='relu'))

    self.model.add(Flatten())
    # self.model.add(Reshape((1, 142)))
    self.model.add(Dense(9, activation=tf.keras.layers.Softmax()))
    
    self.model.summary()

  def train(self):
    self.build_agent(
      self.model,
      self.states
    )
    
    self.agent.compile(
      keras.optimizers.Adam(
        learning_rate=0.0001
      ),
      metrics=['mae']
    )
    self.agent.fit(
      self.tictactoe,
      nb_steps=10,
      visualize=True,
      verbose=1
      #input_shape=(3,3)
    )
      
  def test(self):
    #scores = self.dqn.test(self.tictactoe, nb_episodes=20, visualize=True)
    #print(np.mean(scores.history['episode_reward']))
    pass

  def build_agent(self, model, actions):
    self.policy = BoltzmannQPolicy()
    self.memory = SequentialMemory(
      limit=50000,
      window_length=1
    )
    
    self.agent = DQNAgent(
      model=model,
      memory=self.memory,
      policy=self.policy,
      nb_actions=actions,
      nb_steps_warmup=100,
      target_model_update=1e-2
    )

  def play(self, players, tictactoe):
    table = tictactoe.table.copy()
    action = (0, 0)
    probToWin = 0
    
    # adapt the table to the train data
    if self.idPlayer != 1:
      table = self.swap_id_player_in_table(table, self.idPlayer, 1)
        
    action_predict = self.agent.forward(
      np.array(table).reshape((1 , self.actions)).tolist()[0]
    )
        
    # print(action_predict)
                    
    n_state, reward, done, info = tictactoe.step(action_predict)
    
    self.save(n_state, reward, done, info)
