#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)

import math
import numpy as np
import copy
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, fnet,gnet, args):
        self.game = game
        self.gameCopy= None;
        self.fnet = fnet
        self.gnet = gnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        #self.Vs = {}        # stores game.getValidMoves for board s

    def treeStrategy(self, playerCards, publicHistory, publicCards, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        initialState.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        
        estimOpponentCards= self.gnet.predict(game.getPlayerCards(), game.getpublicHistory(), game.getPublicCards()) # gives a guess of the opponent cards, we can change this to be the actual cards
        for i in range(self.args.numMCTSSims):

			self.gameCopy= copy.deepcopy(self.game)			 #Make another instance of the game for each search
        	gameCopy.setOpponentCard(np.random.choice(gameCopy.getActionSize(),estimOpponentCards)) #choose the opponent cards with a guess

            self.search(estimOpponentCards)

        s = self.game.stringRepresentation() #This is to get a representation of the initial state of the game
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())] #Here you count the number of times that action a was taken in state s

        counts = [x**(1./temp) for x in counts] #Add a temperature factor to emphasize max_a(Nsa) or to sample uniform
        probs = [x/float(sum(counts)) for x in counts] #normalize
        return probs #return pi,tree strategy


    def search(self, estimOpponentCards):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current initialState
        """

        s = self.gameCopy.stringRepresentation() #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary

        if s not in self.Es: # check if s is a known terminal state
            self.Es[s] = self.gameCopy.getGameEnded()
        if self.Es[s]!=0: # This means that the game has terminated (what about this return value though?)
            # terminal node
            return -self.Es[s]

        if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
            # leaf node
            self.Ps[s], v = self.fnet.predict(estimOpponentCards, gameCopy.getPlayerCards(), gameCopy.publicHistory(), gameCopy.publicCards()) #This is begging to get changed for something useful
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
            if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
            else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        action = np.zeros[self.gameCopy.getActionSize(),1] # Encode the action in the one hot format
        action[best_act]=1;

        self.gameCopy.action(action)
        if self.gameCopy.getPlayer() == self.game.getPlayer(): # If somehow it is still your turn, play again, same rules apply 
       
        	v = -self.search(estimOpponentCards)
       
        else:

        	v = self.oppSearch(estimOpponentCards) 			   #If not, let the opponent play their turn. The argument is carried to continue the tree search

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
	return -v

	def oppSearch(estimOpponentCards) 
	#This function is virtually the same as the search method,  
	# but no decisions are made based on Q. Tries to model an opponent with a fixed strategy. 
	# It was written to avoid clunky 'if' statements in the search code
	    
	    s = self.gameCopy.stringRepresentation() #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary

        if s not in self.Es: # check if s is a known terminal state
            self.Es[s] = self.gameCopy.getGameEnded(opponentCards, playerCards, publicHistory, publicCards, 1)
        if self.Es[s]!=0: # This means that the game has terminated (what about this return value though?)
            # terminal node
            return -self.Es[s]

        if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
            # leaf node
            estimPlayerCards = self.gnet.predict(game.getPlayerCards(), game.getpublicHistory(), game.getPublicCards()) 	#Allow the opponent to infer your cards based on your bets 
            self.Ps[s], v = self.fnet.predict(estimPlayerCards, game.getPlayerCards(), game.getpublicHistory(), game.getPublicCards())   #Opponent Strategy. 

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Ns[s] = 0
            return -v

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.gameCopy.getActionSize()):
            if (s,a) in self.Nsa:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
            else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?
            if u > cur_best:
                cur_best = u
                best_act = a

        action = np.zeros[self.gameCopy.getActionSize(),1] # Encode the action in the one hot format
        action[best_act]=1;

        self.gameCopy.action(action) #Execute the action

        if self.gameCopy.getPlayer() == self.game.getPlayer(): # Pass the turn to the original player
       
        	v = self.search(estimOpponentCards)
       
        else:

        	v = -self.oppSearch(estimOpponentCards) 							#If somehow it is still the opponent's turn, let them play again

        if (s,a) in self.Nsa:
            self.Nsa[(s,a)] += 1

        else:
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
	return -v