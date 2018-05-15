#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)

import math
import numpy as np
import copy
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, fnet, gnet, numMCTSSims, cpuct):
        self.game = game
        self.gameCopy= None;
        self.fnet = fnet
        self.gnet = gnet
        self.numMCTSSims=numMCTSSims
        self.cpuct=cpuct
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        #self.Vs = {}        # stores game.getValidMoves for board s

    def cleanTree(self):
    #Clean the temporal variables, so that the same instance of the simulation can be used more than once

        self.Qsa = {}       
        self.Nsa = {}      
        self.Ns = {}        
        self.Ps = {}        
        self.Es = {}



    def treeStrategy(self, playerCards, publicHistory, publicCards, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        initialState.
        The simulation copies an instance of the game and then carries it forward multiple times from two perspectives.
        Returns:
        probs: a policy vector where the probability of the ith action is
        proportional to Nsa[(s,a)]**(1./temp)
        """
        
        estimOpponentCards= self.gnet.predict(self.game.getPlayerCard(), self.game.getPublicHistory(), self.game.getPublicCard()) # gives a guess of the opponent cards, we can change this to be the actual cards
        for i in range(self.numMCTSSims): 
        
        	self.gameCopy= copy.deepcopy(self.game)			 #Make another instance of the game for each search
        	self.gameCopy.setOpponentCard(np.random.choice(self.gameCopy.getActionSize(),estimOpponentCards)) #choose the opponent cards with a guess
        	print(i)
        	self.search(estimOpponentCards)

        s = self.game.stringRepresentation() #This is to get a representation of the initial state of the game
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())] #Here you count the number of times that action a was taken in state s

        counts = [x**(1./temp) for x in counts] #Add a temperature factor to emphasize max_a(Nsa) or to sample uniform
        probs = [x/float(sum(counts)) for x in counts] #normalize
        return probs 		#return pi,tree strategy

    def search(self, estimOpponentCards):

    	if (self.gameCopy.getPlayer() == self.game.getPlayer()):
        	print("Player")
        else:
        	print("opponent")

    	s = self.gameCopy.stringRepresentation() #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary

        if s not in self.Es: # check if s is a known terminal state
            self.Es[s] = self.gameCopy.getOutcome()
        if self.Es[s] is not None: 

            print("Terminal")
            return -self.Es[s][self.game.getPlayer()] 

        if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
            # leaf node
            if self.gameCopy.getPlayer() == self.game.getPlayer(): #original player 
            	self.Ps[s], v = self.fnet.predict(estimOpponentCards, self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard()) #This is begging to get changed for something useful
            else:
				estimPlayerCards = self.gnet.predict(self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard()) 	#Allow the opponent to infer your cards based on your bets 
				self.Ps[s], v = self.fnet.predict(estimPlayerCards, self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard())   #Opponent Strategy.
            
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            #self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.gameCopy.getActionSize()):
            if (s,a) in self.Nsa:
                    u = (self.gameCopy.getPlayer() == self.game.getPlayer())*self.Qsa[(s,a)] + self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
            else:
                    u = self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a=best_act
        action = np.zeros((self.gameCopy.getActionSize(),1)) # Encode the action in the one hot format
        action[a]=1;
        print(action)
        self.gameCopy.action(action)
        v = ((-1)**(self.gameCopy.getPlayer() == self.game.getPlayer()))*self.search(estimOpponentCards)

        if (s,a) in self.Nsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1
        print("Q="+str(self.Qsa[(s,a)]) +"    " +s)

        self.Ns[s] += 1

        return -v