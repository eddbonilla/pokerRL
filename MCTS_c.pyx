#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)

cimport cython
import math
import numpy as np
cimport numpy as np
from leduc_c cimport LeducGame
import copy
import time
EPS = 1e-8


cdef class MCTS():
	"""
	This class handles the MCTS tree.
	"""

	cdef int numMCTSSims,cpuct
	cdef double temp,floor
	cdef dict Qsa,Nsa,Ns,Ps,
	cdef LeducGame game, gameCopy
	cdef object nnets 

	def __init__(self, nnets, int numMCTSSims, int cpuct, int temp = 1, double floor = 0.05):
		self.game = None
		self.gameCopy= None;
		self.nnets = nnets #neural networks used to predict the cards and/or action probabilities
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct
		self.temp = temp
		self.floor = floor
		self.Qsa = {}	   # stores Q values for s,a (as defined in the paper)
		self.Nsa = {}	   # stores #times edge s,a was visited
		self.Ns = {}		# stores #times board s was visited
		self.Ps = {}		# stores initial policy (returned by neural net)


		#self.Es = {}		# stores game.getGameEnded ended for board s
		#self.Vs = {}		# stores game.getValidMoves for board s

	cpdef void reduceTemp(self):
		if self.temp > 0.2:
			self.temp = self.temp/1.001


	cpdef void cleanTree(self):
	#Clean the temporal variables, so that the same instance of the simulation can be used more than once

		self.Qsa = {}	   
		self.Nsa = {}	  
		self.Ns = {}		
		self.Ps = {}		
		#self.Es = {}


	@cython.boundscheck(False)
	@cython.wraparound(False)
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
		cdef np.ndarray estimOpponentCards= self.game.regulariseOpponentEstimate(self.nnets.estimateOpponent(self.game.getPublicHistory(), self.game.getPublicCard())) # gives a guess of the opponent cards, we can change this to be the actual cards
		for i in range(self.numMCTSSims): 
		
			self.gameCopy= self.game.copy()			 #Make another instance of the game for each search
			self.gameCopy.setOpponentCard(np.random.choice(3,p=estimOpponentCards)) #choose the opponent cards with a guess
			#if i%100 == 0: print(i)
			self.search()
			#if i>2: print("N="+str(self.Nsa[(self.game.playerInfoStringRepresentation(),0)]))

		cdef str s = self.game.playerInfoStringRepresentation() #This is to get a representation of the initial state of the game
		cdef np.ndarray counts = self.Nsa[s] #Here you count the number of times that action a was taken in state s

		counts = counts**(1./self.temp) #Add a temperature factor to emphasize max_a(Nsa) or to sample uniform
		cdef np.ndarray treeStrategy = counts /float(np.sum(counts)) #normalize
		#averageStrategy = self.Ps[s]
		return treeStrategy 		#return pi,tree strategy

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double search(self, int exploitSearch = False):

		cdef str s = self.gameCopy.playerInfoStringRepresentation() #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary
		cdef int pot = self.gameCopy.getPot()
		cdef int playerMove = self.gameCopy.getPlayer() == self.game.getPlayer()
		#print("opponent")
		cdef str s_pub = self.gameCopy.publicInfoStringRepresentation()

		cdef int finished = self.gameCopy.isFinished()
		if self.gameCopy.isFinished(): # check if s is a known terminal state

			#print("Terminal")
			#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(self.gameCopy.getOutcome()[self.game.getPlayer()]))
			return self.gameCopy.getOutcome()[self.game.getPlayer()] #Always get outcome for original player


		cdef double value
		if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
			# leaf node
			
			#fnet and gnet integrate into one function 
			self.Ps[s], value = self.nnets.policyValue(self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard())   #Opponent Strategy.

			#self.Vs[s] = valids
			if playerMove:
				self.Ps[s] = self.Ps[s]**(self.temp)
				self.Qsa[s] = value * np.ones(3)
				self.Ns[s] = 0
				self.Nsa[s] = np.zeros(3)
				return (value*pot)
				#if exploitSearch:
				#	self.Ps[s] = np.ones(3)/3
			elif s_pub not in self.Ns:
				self.Ns[s_pub] = 0
				self.Nsa[s_pub] = np.zeros(3)

		# pick the action with the highest upper confidence bound
		cdef np.ndarray u 
		if playerMove:
			u = self.Qsa[s] + np.sqrt(self.Ns[s]+EPS)*self.cpuct*(self.Ps[s]+self.floor)/(1+self.Nsa[s])

		else:
			u = self.cpuct*self.Ps[s]*math.sqrt(self.Ns[s_pub]+EPS)/(1+self.Nsa[s_pub])

		#print(u)
		cdef int a=np.argmax(u)
		#print("probs =" +str(self.Ps[s])+", playerMove = "+str(playerMove)+ ", action ="+str(a))


		cdef int bet = self.gameCopy.action(action=a)
		cdef double net_winnings = -bet*(playerMove) + self.search(exploitSearch = exploitSearch)
		cdef double v = net_winnings/pot

		if playerMove:
			self.Qsa[s][a] = float(self.Nsa[s][a]*self.Qsa[s][a] + v)/(self.Nsa[s][a]+1)
			self.Nsa[s][a] += 1
			self.Ns[s] += 1
		else:
			self.Ns[s_pub] += 1
			self.Nsa[s_pub][a] += 1
		#print("Q="+str(self.Qsa[(s,a)]))
		#print("net_winnings=" +str(net_winnings))
		
		#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(net_winnings)+", dealer =" + str(self.gameCopy.dealer))

		return net_winnings

	def setNumSimulations(self,int newNumMCTSSims):
		self.numMCTSSims=newNumMCTSSims

	def findExploitability(self, int numExploitSims = 5000 ):
		start = time.time()
		#if searchBlind:
		#	self.cleanTree()

		#winnings = 0
		
		self.game = LeducGame()
		for i in range(numExploitSims):
			self.game.resetGame()
			self.gameCopy= self.game.copy()
			#print(self.gameCopy.playerInfoStringRepresentation())
			#input("OK?")
			self.game.setPlayer(0)
			self.search()
		end = time.time()
		print("Exploitability calculation time: "+str(end - start))
		#print("Average Winnings: " + str(winnings/numExploitSims))
		#print(self.Qsa)
		cdef double exploitability = -1
		for card in range(3):
			history = np.zeros((2,2,3,2),dtype=int)
			publicCard = np.zeros(3,dtype=int)
			print(self.Qsa['0'+str(card)+str(history)+str(publicCard)])
			#print(np.max(self.Qsa[str(0)+str(card)+str(history)+str(publicCard)]))
			exploitability += np.max(self.Qsa[str(0)+str(card)+str(history)+str(publicCard)])/3.
			for oppCard in range(3):
				p = self.Ps['1'+str(oppCard)+str(history)+str(publicCard)]
				history[1,0,0,0] = 1
				raiseString = '0'+str(card)+str(history)+str(publicCard)
				history[1,0,0,0] = 0
				history[1,0,0,1] = 1
				checkString = '0'+str(card)+str(history)+ str(publicCard)
				history[1,0,0,1] = 0
				if raiseString in self.Qsa:
					exploitability += p[0]*np.max(self.Qsa[raiseString])/9.
				if checkString in self.Qsa:
					exploitability +=p[1]*np.max(self.Qsa[checkString])/9.
					exploitability +=p[2]/9.
		return exploitability

