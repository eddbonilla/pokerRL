#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)

import math
import numpy as np
from leduc import LeducGame
import copy
import time
EPS = 1e-8

class MCTS():
	"""
	This class handles the MCTS tree.
	"""

	def __init__(self, nnets, numMCTSSims, cpuct, temp = 1):
		self.game = None
		self.gameCopy= None;
		self.nnets = nnets #neural networks used to predict the cards and/or action probabilities
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct
		self.temp = temp
		self.Qsa = {}	   # stores Q values for s,a (as defined in the paper)
		self.Nsa = {}	   # stores #times edge s,a was visited
		self.Ns = {}		# stores #times board s was visited
		self.Ps = {}		# stores initial policy (returned by neural net)


		self.Es = {}		# stores game.getGameEnded ended for board s
		#self.Vs = {}		# stores game.getValidMoves for board s

		#Harcoded Parameters. To be deprecated soon
		self.tempDecayRate = 1.02
		self.treeSimAdditionRate =5

	def reduceTempAndAddSearches(self):
		self.temp = self.temp/self.tempDecayRate
		self.numMCTSSims += self.treeSimAdditionRate

	def cleanTree(self):
	#Clean the temporal variables, so that the same instance of the simulation can be used more than once

		self.Qsa = {}	   
		self.Nsa = {}	  
		self.Ns = {}		
		self.Ps = {}		
		self.Es = {}



	def strategy(self,game):
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
			self.gameCopy.setOpponentCard(np.random.choice(int(self.gameCopy.params["actionSize"]),p=estimOpponentCards)) #choose the opponent cards with a guess
			#if i%100 == 0: print(i)
			self.search()
			#if i>2: print("N="+str(self.Nsa[(self.game.playerInfoStringRepresentation(),0)]))

		s = self.game.playerInfoStringRepresentation() #This is to get a representation of the initial state of the game
		counts = self.Nsa[s] #Here you count the number of times that action a was taken in state s

		counts = counts**(1./self.temp) #Add a temperature factor to emphasize max_a(Nsa) or to sample uniform
		treeStrategy = counts /float(np.sum(counts)) #normalize
		averageStrategy = self.Ps[s]
		return averageStrategy, treeStrategy 		#return pi,tree strategy

	def search(self, exploitSearch = False):

		s = self.gameCopy.playerInfoStringRepresentation() #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary
		pot = self.gameCopy.getPot()
		playerMove = self.gameCopy.getPlayer() == self.game.getPlayer()
		#print("opponent")
		s_pub = self.gameCopy.publicInfoStringRepresentation()

		if self.gameCopy.isFinished(): # check if s is a known terminal state

			#print("Terminal")
			#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(self.gameCopy.getOutcome()[self.game.getPlayer()]))
			return self.gameCopy.getOutcome()[self.game.getPlayer()] #Always get outcome for original player

		if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
			# leaf node
			
			#fnet and gnet integrate into one function 
			self.Ps[s], v = self.nnets.policyValue(self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard())   #Opponent Strategy.

			#self.Vs[s] = valids
			if playerMove:
				self.Qsa[s] = v * np.ones(self.game.params["actionSize"])
				self.Ns[s] = 0
				self.Nsa[s] = np.zeros(self.game.params["actionSize"])
				#if exploitSearch:
				#	self.Ps[s] = np.ones(3)/3
			elif s_pub not in self.Ns:
				self.Ns[s_pub] = 0
				self.Nsa[s_pub] = np.zeros(self.game.params["actionSize"])

			#if not exploitSearch:
			return pot*(1-v +(2*v-1)*playerMove)

		# pick the action with the highest upper confidence bound

		if playerMove:
			u = self.Qsa[s] + math.sqrt(self.Ns[s]+EPS)*self.cpuct*self.Ps[s]/(1+self.Nsa[s])

		else:
			u = self.cpuct*self.Ps[s]*math.sqrt(self.Ns[s_pub]+EPS)/(1+self.Nsa[s_pub])

		#print(u)
		a=np.argmax(u)
		#print("probs =" +str(self.Ps[s])+", playerMove = "+str(playerMove)+ ", action ="+str(a))

		_,bet = self.gameCopy.action(action=a)
		net_winnings = -bet*(playerMove) + self.search(exploitSearch = exploitSearch)
		v = net_winnings/pot

		if playerMove:
			self.Qsa[s][a] = float (self.Nsa[s][a]*self.Qsa[s][a] + v)/(self.Nsa[s][a]+1)
			self.Nsa[s][a] += 1
			self.Ns[s] += 1
		else:
			self.Ns[s_pub] += 1
			self.Nsa[s_pub][a] += 1
		#print("Q="+str(self.Qsa[(s,a)]))
		#print("net_winnings=" +str(net_winnings))
		
		#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(net_winnings)+", dealer =" + str(self.gameCopy.dealer))

		return net_winnings

	def setNumSimulations(self,newNumMCTSSims):
		self.numMCTSSims=numMCTSSims

	def findExploitability(self, numExploitSims = 5000 ):
		start = time.time()
		#if searchBlind:
		#	self.cleanTree()

		#winnings = 0
		
		self.game = LeducGame() #explicit reference to leduc
		for i in range(numExploitSims):
			self.game.resetGame()
			self.gameCopy= copy.deepcopy(self.game)
			self.game.setPlayer(0)
			self.search()
		end = time.time()
		print("Exploitability calculation time: "+str(end - start))
		#print("Average Winnings: " + str(winnings/numExploitSims))
		exploitability = -1
		for card in range(3):
			history = np.zeros((2,2,3,2))
			publicCard = np.zeros(3)
			print(self.Qsa['0'+str(card)+str(history)+str(publicCard)])
			print("max= " +str(np.max(self.Qsa['0'+str(card)+str(history)+str(publicCard)])))
			exploitability += 1/3*np.max(self.Qsa['0'+str(card)+str(history)+str(publicCard)])
			print(exploitability)
			for oppCard in range(3):
				p = self.Ps['1'+str(oppCard)+str(history)+str(publicCard)]
				history[1,0,0,0] = 1
				raiseString = '0'+str(card)+str(history)+str(publicCard)
				history[1,0,0,0] = 0
				history[1,0,0,1] = 1
				checkString = '0'+str(card)+str(history)+ str(publicCard)
				history[1,0,0,1] = 0
				if raiseString in self.Qsa:
					exploitability += 1/9*p[0]*np.max(self.Qsa[raiseString])
				if checkString in self.Qsa:
					exploitability +=1/9*p[1]*np.max(self.Qsa[checkString])
					exploitability +=1/9*p[2]
					print(self.Qsa[checkString])
		return exploitability
	def setTreeSearchParams(params):
		self.numMCTSSims=params["initialNumMCTS"]
		self.temp=params["initialTreeTemperature"]
		self.tempDecayRate=params["tempDeacyRate"]
		self.treeSimAdditionRate=params["treeSimAdditionRate"]
		self.cpuct=params["cpuct"]

