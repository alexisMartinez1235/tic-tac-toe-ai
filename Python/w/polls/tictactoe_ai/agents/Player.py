

class Player(object):
    def __init__(self, idPlayer, name: str):
        if name == "":
              self.name = input("Player insert your name:")
        self.name = name
        self.idPlayer = idPlayer
        self.n_state = None
        self.reward = None
        self.done = False
        self.info = None
        
    def play(self, players, tictactoe, action=None):
        field = 0
        column = 0
        if not self.done:
            if action == None:
                field = int(input('Enter field:'))
                column = int(input('Enter column:'))
            else:
                field = action[0]
                cols = action[1]
        n_state, reward, done, info = tictactoe.step((field, column))

        self.save(n_state, reward, done, info)
    
    def save(self, n_state, reward, done, info):
        self.n_state = n_state
        self.reward = reward
        self.done = done
        self.info = info
        
    def showResults(self):
        return self.n_state, self.reward, self.done, self.info
    
    def showDescription(self, redirectPrintTo):
        print('id:{} State:{} Reward:{} Done:{} Info:{}'.format(self.idPlayer, self.n_state, self.reward, self.done, self.info), file=redirectPrintTo)

    def getPlayer(self):
        return self