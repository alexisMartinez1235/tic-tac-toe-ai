from agents.Player import Player
import numpy as np

class AI_Reinforcement_Random(Player):
    def __init__(self, idPlayer, name: str):
        super().__init__(idPlayer, name)
        self.score = 0
        # self.episode=1

    def play(self, players, tictactoe):
        # tictactoe.render(players)

        action = tictactoe.action_space.sample()

        n_state, reward, done, info = tictactoe.step(action)
        
        self.save(n_state, reward, done, info)

    def showDescription(self, redirectPrintTo):
        print('id:{} Episode:{} Reward:{} Info:{}'.format(self.idPlayer, 1, self.reward, self.info), file=redirectPrintTo)
        
    def swap_id_player_in_table(self, arr, idplayerWinner, idPlayerLooser):
        arrAux = np.matrix(arr)
        
        np.place(arrAux, arrAux == idplayerWinner, -idplayerWinner)
        np.place(arrAux, arrAux == idPlayerLooser, idplayerWinner)
        np.place(arrAux, arrAux == -idplayerWinner, idPlayerLooser)
    
        return arrAux

