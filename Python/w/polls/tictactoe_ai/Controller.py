import numpy as np
import matplotlib.pyplot as plt
import os

from environment.TicTacToe import TicTacToe

def playTwoPlayers(
        player1,
        player2,
        allowDebug=False,
        # start at one because 0 represent an empty cell and should be positive integers 
        tictactoe = TicTacToe(2, np.zeros((3, 3), np.int32), 3, False)
    ):
    
    tictactoe.redirectToChange(allowDebug)
    #tictactoe = TicTacToe(2, np.array([
    #    [2, 1, 2],
    #    [1, 1, 0],
    #    [2, 2, 1]
    #], np.int32), 3, allowDebug)
    tictactoe.add_players([player1, player2])
    tictactoe.reset()
    tictactoe.run_episode()
    
    return [ tictactoe.history, tictactoe.winner]



def createPlot(historyAndWinner, winnerAtitle, winnerBtitle, redirect=open(os.devnull, 'w'), num_of_games=300):
    batch = 10
    number_that_divides_list = num_of_games / batch

    winnerAList = np.zeros(shape=batch, dtype=float)
    drawList = np.zeros(shape=batch, dtype=float)
    winnerBList = np.zeros(shape=batch, dtype=float)

    num_played = 0

    winnerA= 0
    draw = 0
    winnerB= 0

    for i in range(0, num_of_games):
        history, winner = historyAndWinner()
        
        # add probabilities points to winnerA, draw and winnerB 
        if winner == 1:
            winnerA = winnerA + 1
        elif winner == 2:
            winnerB = winnerB + 1
        else:
            draw = draw + 1

        # save proxination
        if num_played % number_that_divides_list == 0 and num_played != 0:
            pos = int(num_played/number_that_divides_list)

            winnerAList.itemset(pos, winnerA/num_played)
            drawList.itemset(pos, draw/num_played)
            winnerBList.itemset(pos, winnerB/num_played)

            print(str(num_played) + "/" + str(num_of_games) + " games", file=redirect)
        num_played = num_played + 1
    
    
    x = np.linspace(batch, num_of_games, batch)
    y= np.ones(batch)
    
    plt.figure(figsize=(13, 9))
    plt.title("Probabilities for " + winnerAtitle + " vs " + winnerBtitle)
    
    plt.xlabel("games") 
    plt.ylabel("probability")
    
    plt.plot(x, winnerAList, 'r', label=winnerAtitle)
    plt.plot(x, drawList, 'g', label="draw")
    plt.plot(x, winnerBList, 'b', label=winnerBtitle)
    
    plt.plot(x, winnerAList[batch-1]*y, 'r', label='limit ' + winnerAtitle + ': ' + str(winnerAList[batch-1]))
    plt.plot(x, drawList[batch-1]*y, 'g', label='limit draw ' + str(drawList[batch-1]))
    plt.plot(x, winnerBList[batch-1]*y, 'b', label='limit ' + winnerBtitle + ': ' + str(winnerBList[batch-1]))
    
    plt.legend(loc='upper left')
    
    plt.show()

if __name__ == "main":
      
  # createPlot(historyAndWinner=playTwoPlayers, title="AI Random vs Random")
  # createPlot(historyAndWinner=playAIRandomVSReinforcement, title="AI Random vs AI_Reinforcement_Random")
  # createPlot(historyAndWinner=playReinforcementVSReinforcement, title="AI_Reinforcement_Random vs AI_Reinforcement_Random")
  pass