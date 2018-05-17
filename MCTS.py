#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)

import math
import numpy as np
import copy
EPS = 1e-8

class MCTS():
	"""
	This class handles the MCTS tree.
	"""

	def __init__(self, nnets, numMCTSSims, cpuct):
		self.game = None
		self.gameCopy= None;
		self.nnets = nnets #neural networks used to predict the cards and/or action probabilities
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct
		self.Qsa = {}	   # stores Q values for s,a (as defined in the paper)
		self.Nsa = {}	   # stores #times edge s,a was visited
		self.Ns = {}		# stores #times board s was visited
		self.Ps = {}		# stores initial policy (returned by neural net)

		self.Es = {}		# stores game.getGameEnded ended for board s
		#self.Vs = {}		# stores game.getValidMoves for board s

	def cleanTree(self):
	#Clean the temporal variables, so that the same instance of the simulation can be used more than once

		self.Qsa = {}	   
		self.Nsa = {}	  
		self.Ns = {}		
		self.Ps = {}		
		self.Es = {}



	def strategy(self,game,temp=1):
		"""
		This function performs numMCTSSims simulations of MCTS starting from
		initialState.
		The simulation copies an instance of the game and then carries it forward multiple times from two perspectives.
		Returns:
		probs: a policy vector where the probability of the ith action is
		proportional to Nsa[(s,a)]**(1./temp)
		"""
		self.game=game
		estimOpponentCards= self.nnets.estimateOpponent(self.game.getPlayerCard(), self.game.getPublicHistory(), self.game.getPublicCard()) # gives a guess of the opponent cards, we can change this to be the actual cards
		for i in range(self.numMCTSSims): 
		
			self.gameCopy= copy.deepcopy(self.game)			 #Make another instance of the game for each search
			self.gameCopy.setOpponentCard(np.random.choice(self.gameCopy.params["actionSize"],estimOpponentCards)) #choose the opponent cards with a guess
			if i%100 == 0: print(i)
			self.search()
			#if i>2: print("N="+str(self.Nsa[(self.game.playerInfoStringRepresentation(),0)]))

		s = self.game.playerInfoStringRepresentation() #This is to get a representation of the initial state of the game
		counts = np.array([self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.params["actionSize"])]) #Here you count the number of times that action a was taken in state s

		counts = counts**(1./temp) #Add a temperature factor to emphasize max_a(Nsa) or to sample uniform
		treeStrategy = counts /float(np.sum(counts)) #normalize
		averageStrategy = self.Ps[s]
		return averageStrategy, treeStrategy 		#return pi,tree strategy

	def search(self):

		s = self.gameCopy.playerInfoStringRepresentation() #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary
		pot = self.gameCopy.getPot()
		playerMove = self.gameCopy.getPlayer() == self.game.getPlayer()

		if playerMove:
			#print("Player")
			sForN = s
		else:
			#print("opponent")
			sForN = self.gameCopy.publicInfoStringRepresentation()

		if self.gameCopy.isFinished(): # check if s is a known terminal state

			#print("Terminal")
			return self.gameCopy.getOutcome()[self.game.getPlayer()] #Always get outcome for original player

		if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
			# leaf node
			
			#fnet and gnet integrate into one function - I just thinkthis might be slightly faster/cleaner -G 
			self.Ps[s], v = self.nnets.policyValue(self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard())   #Opponent Strategy.

			#self.Vs[s] = valids
			self.Ns[sForN] = 0
			return pot*(1-v +(2*v-1)*playerMove)

		cur_best = -float("inf")
		best_act = -1

		# pick the action with the highest upper confidence bound
		for a in range(self.gameCopy.params["actionSize"]):
			if (sForN,a) in self.Nsa:
					u = (playerMove)*self.Qsa[(s,a)] + self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[sForN])/(1+self.Nsa[(sForN,a)])
			else:
					u = self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[sForN] + EPS)     # Q = 0 ?

			if u > cur_best:
				cur_best = u
				best_act = a

		a=best_act
		action = np.zeros((self.gameCopy.params["actionSize"],1)) # Encode the action in the one hot format
		action[a]=1;
		#print(action)
		_,bet = self.gameCopy.action(action)
		net_winnings = -bet*(playerMove) + self.search()
		v = net_winnings/pot
		if (s,a) in self.Nsa:
			self.Qsa[(s,a)] = float (self.Nsa[(sForN,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(sForN,a)]+1)
			self.Nsa[(sForN,a)] += 1

		else:
			self.Qsa[(s,a)] = v
			self.Nsa[(sForN,a)] = 1
		#print("Q="+str(self.Qsa[(s,a)]))
		#print("net_winnings=" +str(net_winnings))
		self.Ns[sForN] += 1
		return net_winnings