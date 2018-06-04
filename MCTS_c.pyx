#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)

cimport cython
import math
import numpy as np
cimport numpy as np
from leduc_c cimport LeducGame
from holdEm_c cimport HoldEmGame
import copy
import time
EPS = 1e-8


cdef class MCTS():
	"""
	This class handles the MCTS tree.
	"""

	cdef int numMCTSSims,cpuct
	cdef double temp,floor,tempDecayRate
	cdef dict Qsa,Nsa,Ns,Ps,
	cdef Game game, gameCopy
	cdef object nnets 

	def __init__(self, nnets, int numMCTSSims, int cpuct, int temp = 1, double floor = 0.05, double tempDecayRate = 1.0005,bool holdEm = False):
		self.game = None
		self.gameCopy= None;
		self.nnets = nnets #neural networks used to predict the cards and/or action probabilities
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct
		self.temp = temp
		self.tempDecayRate =tempDecayRate
		self.floor = floor
		self.holdEm = holdEm
		self.Qsa = {}	   # stores Q values for s,a (as defined in the paper)
		self.Nsa = {}	   # stores #times edge s,a was visited
		self.Ns = {}		# stores #times board s was visited
		self.Ps = {}		# stores initial policy (returned by neural net)


		#self.Es = {}		# stores game.getGameEnded ended for board s
		#self.Vs = {}		# stores game.getValidMoves for board s

	cpdef void reduceTemp(self):
		if self.temp > 0.1:
			self.temp = self.temp/self.tempDecayRate

	def increaseNumSimulations(self):
		self.numMCTSSims+=1

	def setTreeSearchParams(self,params):
		self.numMCTSSims=params["initialNumMCTS"]
		self.temp=params["initialTreeTemperature"]
		self.tempDecayRate=params["tempDecayRate"]
		self.cpuct=params["cpuct"]


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
			self.search(root = True)
			#if i>2: print("N="+str(self.Nsa[(self.game.playerInfoStringRepresentation(),0)]))

		cdef str s = self.game.playerInfoStringRepresentation() #This is to get a representation of the initial state of the game
		cdef np.ndarray counts = self.Nsa[s] #Here you count the number of times that action a was taken in state s

		counts = counts**(1./self.temp) #Add a temperature factor to emphasize max_a(Nsa) or to sample uniform
		cdef np.ndarray treeStrategy = counts /float(np.sum(counts)) #normalize
		#averageStrategy = self.Ps[s]
		return treeStrategy 		#return pi,tree strategy

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double search(self, int root = False):

		cdef str s = self.gameCopy.playerInfoStringRepresentation() #gives a code for the state of the game, it is a unique string of characters that can be placed in a python dictionary
		cdef int pot = self.gameCopy.getPot()
		cdef int playerMove = self.gameCopy.getPlayer() == self.game.getPlayer()
		#print("opponent")
		
		if self.gameCopy.isFinished(): # check if s is a known terminal state

			#print("Terminal")
			#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(self.gameCopy.getOutcome()[self.game.getPlayer()]))
			return self.gameCopy.getOutcome()[self.game.getPlayer()] #Always get outcome for original player

		cdef np.ndarray strategy
		cdef double value
		if not playerMove:
			strategy,value = self.nnets.policyValue(self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard())
			pFold = strategy[2]
			strategy[2] = 0
			pot = self.gameCopy.getPot()
			self.gameCopy.action(action = -1, strategy = strategy/np.sum(strategy))
			return pFold*pot + (1 - pFold)*self.search()


		if s not in self.Ps: #Have we been on this state during the search? if yes, then no need to reevaluate it
			# leaf node
			
			#fnet and gnet integrate into one function 
			self.Ps[s], value = self.nnets.policyValue(self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard())   #Opponent Strategy.

			#self.Vs[s] = valids
			
			self.Ps[s] = self.Ps[s]**(self.temp)
			self.Qsa[s] = value * np.ones(3)
			self.Ns[s] = 0
			self.Nsa[s] = np.zeros(3)
			return (value*pot)
			#if exploitSearch:
			#	self.Ps[s] = np.ones(3)/3

		# pick the action with the highest upper confidence bound
		cdef np.ndarray u
		if not root:
			u = self.Qsa[s] + np.sqrt(self.Ns[s]+EPS)*self.cpuct*(self.Ps[s])/(1+self.Nsa[s])
		else:
			u = self.Qsa[s] + np.sqrt(self.Ns[s]+EPS)*self.cpuct*(self.Ps[s]**self.temp)/(1+self.Nsa[s])
		cdef int a=np.argmax(u)
		#print("probs =" +str(self.Ps[s])+", playerMove = "+str(playerMove)+ ", action ="+str(a))


		cdef int bet = self.gameCopy.action(action=a)
		cdef double net_winnings = -bet*(playerMove) + self.search()
		cdef double v = net_winnings/pot

		self.Qsa[s][a] = float(self.Nsa[s][a]*self.Qsa[s][a] + v)/(self.Nsa[s][a]+1)
		self.Nsa[s][a] += 1
		self.Ns[s] += 1
		
		#print("Q="+str(self.Qsa[(s,a)]))
		#print("net_winnings=" +str(net_winnings))
		
		#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(net_winnings)+", dealer =" + str(self.gameCopy.dealer))

		return net_winnings

	def exploitabilitySearch(self,localGame,belief,prevGame,prevAction): #it returns net winnings for exploiting strategy by doing an exhaustive search
		if prevGame.getRound() is not localGame.getRound(): #time to set the public card (or something)
			#Ppub=np.sum(belief,axis=1) #marginalize over the opponent card ##
			Ppub=np.sum(belief,axis=2) ##for pcard inside the mix
			Qsc=np.zeros((3,3)) #numcards,1
			for pubCard in range(3):
				#print("public")
				nextGame=localGame.copy() #copy the current game
				nextGame.setPublicCard(pubCard)
				updatedBelief=np.transpose(np.multiply(np.transpose(belief,axes=[0,2,1]),nextGame.getPublicCard()),axes=[0,2,1])
				##updatedBelief=np.multiply(belief.T,nextGame.publicCardArray).T ##
				summ=np.sum(updatedBelief,axis=(1,2))
				for pCard in range(3):
					if summ[pCard]!=0:
						updatedBelief[pCard,:,:]=updatedBelief[pCard,:,:]/summ[pCard]
					##updatedBelief=updatedBelief/np.sum(updatedBelief) ##
				Qsc[:,pubCard]=np.reshape(self.exploitabilitySearch(nextGame,belief=updatedBelief,prevGame=nextGame,prevAction=prevAction),3)
			
			return np.diag(np.dot(Qsc,Ppub.T))
		else:

			exploitingPlayer= (localGame.getPlayer()==self.game.getPlayer())
			#if state is terminal return the value
			if localGame.isFinished():
				Vs=np.zeros((3,1),dtype=float) #explicit reference to number of cards
				for pCard in range(3):
					for oppCard in range(3):
						nextGame=prevGame.copy()
						if nextGame.getPlayer()==self.game.getPlayer():
							nextGame.setOpponentCard(oppCard)
							nextGame.setPlayerCard(pCard)
						else:
							nextGame.setPlayerCard(oppCard)
							nextGame.setOpponentCard(pCard)
						nextGame.action(action=prevAction)
						vTerm=nextGame.getOutcome()[self.game.getPlayer()]
						Vs[pCard]+=vTerm*(np.sum(belief[pCard,:,:],axis=0)[oppCard])
				return Vs #Always get outcome for exploiting player
				#initialize values and keep track of bets or something
			numActions=3
			Qsa=np.zeros((3,numActions)) #explicit reference to the number of cards

			if not exploitingPlayer:  #if the strategy player plays return p.v (fut)
				#print("exploited")
				#input("OK?")
				#print(localGame.getPublicHistory())
				#print(localGame.isFinished())
				Ps=np.zeros((3,numActions)) #Explicit reference to number of cards in leduc
				for oppCard in range(3): #explicit reference to number of cards in leduc
					localGame.setPlayerCard(oppCard)
	
					Ps[oppCard,:],_ = self.nnets.policyValue(localGame.getPlayerCard(), localGame.getPublicHistory(), localGame.getPublicCard()) #get probabilities for each card, do not browse the dict, it is slower for some reason -E
					#s = localGame.playerInfoStringRepresentation() #We cannot use the information because it has been raised to the Temp
					#if s not in self.Ps: #
					#Ps[oppCard]=self.Ps[s]
					#Ps[oppCard,:]=[0.5,0.5,0.]
				#print("belief = "+str(belief))
				Pa=np.dot(np.sum(belief,axis=1),Ps) #marginalize over public cards, axis 1. Probability of taking an action a 
				#print(Pa)
				for a in range(numActions): #Make copies of the game with each action
					if Pa[:,a].any() !=0: #if the action has any probability of happening
						nextGame=localGame.copy()
						_=nextGame.action(action=a)
						#apply bayes rule for updating the beliefs
						updatedBelief=np.multiply(belief,Ps[:,a]) #Probs of taking an action given the opponent cards, not normalized
						summ=np.sum(updatedBelief,axis=(1,2))
						for pCard in range(3):
							if summ[pCard]!=0:
								updatedBelief[pCard,:,:]=updatedBelief[pCard,:,:]/summ[pCard]
						Qsa[:,a]=np.reshape(self.exploitabilitySearch(nextGame,belief=updatedBelief,prevGame=localGame,prevAction=a),3)#or something like that
				return np.diag(np.dot(Qsa,Pa.T))

			else:#if the exploiting player plays return max(v), they only take optimal actions. this also takes into account the bet made at the time
				#print("exploiting")
				#print(localGame.getPublicHistory())
				bets=np.zeros((1,numActions))
				cardV=np.zeros(3)
				for a in range(numActions):
					nextGame=localGame.copy()
					bets[:,a]=nextGame.action(action=a)
					Qsa[:,a]=np.reshape(self.exploitabilitySearch(nextGame,belief,prevGame=localGame,prevAction=a),3)
				
				Vs=Qsa-bets
				return  np.maximum(np.maximum(Vs[:,0],Vs[:,1]),Vs[:,2])#take the max over actions of Vs


	def findAnalyticalExploitability(self):

		start = time.time()
		self.game = LeducGame() #explicit reference to leduc
		numPlayers=2 #2 player game
		numCards=3 #Jack, Queen, King
		cardCopies=2
		numActions=3
		exploitability=-1.
		exploitingPlayerId=1; #Id of the player that is exploited
		self.game.resetGame()

		self.gameCopy=self.game.copy()#copy a fresh game

		#create an array with the conditional probabilities of being in a state
		belief=np.zeros((numCards,numCards,numCards),dtype=float)
		for pCard in range(numCards): #take a card
			Qsa=np.zeros((numActions,1),dtype=float)
			for oppCard in range(numCards):
				for pubCard in range(numCards):
					belief[pCard,pubCard,oppCard]=1.*(2-(oppCard==pCard))*(2-(pubCard==oppCard)-(pubCard==pCard))/20.

		for exploitingPlayer in range(2): #two players
			self.game.setPlayer(exploitingPlayer)
			#start the hunt
			Vs=self.exploitabilitySearch(self.gameCopy.copy(),belief=belief,prevGame=self.gameCopy,prevAction=-1)
			exploitability+=(1./3)*(np.sum(Vs)/2) #3 cards, two starting positions

		end = time.time()
		print("Exploitability calculation time: "+str(end - start))
		return exploitability



