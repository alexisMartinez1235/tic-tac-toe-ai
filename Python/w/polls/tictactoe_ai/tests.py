import numpy as np
from Controller import createPlot, playTwoPlayers

from agents.AI_Reinforcement_Random import AI_Reinforcement_Random
from agents.AIBestDecision import AIBestDecision
from agents.AgentQLearning import AgentQLearning
from agents.Player import Player 

import sys
from environment.TicTacToe import TicTacToe

def test_agent_q_learning():
  tictactoe = TicTacToe(2, np.zeros((3, 3), np.int32), 3, False)

  history, winner = playTwoPlayers(
      AgentQLearning(1, "AgentQLearning", tictactoe),  # type: ignore
      AI_Reinforcement_Random(2, "AI_Reinforcement_Random"),
      tictactoe=tictactoe,
      allowDebug=True,
  )
  tictactoe = TicTacToe(2, np.zeros((3, 3), np.int32), 3, False)
  createPlot(
      lambda: playTwoPlayers(
          AgentQLearning(1, "AgentQLearning", tictactoe),# type: ignore
          AI_Reinforcement_Random(2, "AI_Reinforcement_Random"),
          tictactoe=tictactoe,
          allowDebug=False
      ),
      "AgentQLearning",
      "AI_Reinforcement_Random",
      redirect=sys.stdout,  # type: ignore
      num_of_games=200
  )

def test_ai_reinforcement_random():
  createPlot(
    lambda: playTwoPlayers(
        AI_Reinforcement_Random(1, "AI_Reinforcement_Random"),
        AI_Reinforcement_Random(2, "AI_Reinforcement_Random"),   
    ),
    "AI_Reinforcement_Random",
    "AI_Reinforcement_Random",
    redirect=sys.stdout, # type: ignore
    num_of_games=130
  )
  history, winner = playTwoPlayers(
      AI_Reinforcement_Random(1, "AI_Reinforcement_Random1"),
      AI_Reinforcement_Random(2, "AI_Reinforcement_Random2"),
      allowDebug=True
  )
  try:
      playTwoPlayers(
          Player(1, "Me"),
          AI_Reinforcement_Random(2, "Reinforcement"),
          allowDebug=True

      )
  except:
      print("input error")

def test_ai_best_decision():
  player_best_decision = AIBestDecision(1, "Best decision")
  history, winner = playTwoPlayers(
    player_best_decision, # type: ignore
    AI_Reinforcement_Random(2, "Reforce")
  )
  # plot_model(AIBestDecision.network, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  # Image("model_plot.png")

  # table = np.matrix([0], dtype=np.int32)
  result = np.matrix([[1,0,0],[1,0,0],[2,2,0]], dtype=np.int32)  # type: ignore
  result = np.matrix([[0,0,0],[0,0,0],[0,0,0]], dtype=np.int32)  # type: ignore

  predict = player_best_decision.network.predict(np.array(result).reshape((1, 3, 3))).reshape((3, 3)) # type: ignore
  print(predict)
  history, winner = playTwoPlayers(
      AI_Reinforcement_Random(1, "Reforce"),
      AIBestDecision(2, "Best decision") # type: ignore
  )
  print(history, winner)
  createPlot(
      lambda: playTwoPlayers(
          AIBestDecision(1, "Best decision"), # type: ignore
          AI_Reinforcement_Random(2, "AI_Reinforcement_Random"),
      ),
      "Best decision",
      "AI_Reinforcement_Random",
      redirect=sys.stdout, # type: ignore
      num_of_games=200
  )

  history, winner = playTwoPlayers(
      AIBestDecision(1, "Best decision 1"), # type: ignore
      AIBestDecision(2, "Best decision 2"), # type: ignore
  )
  print(history)

  createPlot(
      lambda: playTwoPlayers(
          AIBestDecision(1, "Best decision1"), # type: ignore
          AIBestDecision(2, "Best decision2"), # type: ignore
      ),
      "Best decision",
      "Best decision",
      redirect=sys.stdout, # type: ignore
      num_of_games=120
  )
  try:
    history, winner = playTwoPlayers(
        AIBestDecision(1, "Best decision 1"), # type: ignore
        AIBestDecision(2, "Best decision 2"), # type: ignore
        allowDebug=True
    )
  except:
      print("input error")