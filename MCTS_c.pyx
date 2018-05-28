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
		
		if self.gameCopy.isFinished(): # check if s is a known terminal state

			#print("Terminal")
			#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(self.gameCopy.getOutcome()[self.game.getPlayer()]))
			return self.gameCopy.getOutcome()[self.game.getPlayer()] #Always get outcome for original player

		cdef np.ndarray strategy
		cdef double value
		if not playerMove:
			strategy,_ = self.nnets.policyValue(self.gameCopy.getPlayerCard(), self.gameCopy.getPublicHistory(), self.gameCopy.getPublicCard())
			return self.gameCopy.action(action = -1, strategy = strategy)

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
		cdef np.ndarray u = self.Qsa[s] + np.sqrt(self.Ns[s]+EPS)*self.cpuct*(self.Ps[s]+self.floor)/(1+self.Nsa[s])
		cdef int a=np.argmax(u)
		#print("probs =" +str(self.Ps[s])+", playerMove = "+str(playerMove)+ ", action ="+str(a))


		cdef int bet = self.gameCopy.action(action=a)
		cdef double net_winnings = -bet*(playerMove) + self.search(exploitSearch = exploitSearch)
		cdef double v = net_winnings/pot

		self.Qsa[s][a] = float(self.Nsa[s][a]*self.Qsa[s][a] + v)/(self.Nsa[s][a]+1)
		self.Nsa[s][a] += 1
		self.Ns[s] += 1
		
		#print("Q="+str(self.Qsa[(s,a)]))
		#print("net_winnings=" +str(net_winnings))
		
		#input("player card =" + str(self.game.getPlayerCard()) + ", opponent card ="+str(self.game.getOpponentCard())+", public card ="+str(self.gameCopy.getPublicCard())+ ", net winnings = "+str(net_winnings)+", dealer =" + str(self.gameCopy.dealer))

		return net_winnings

	def setNumSimulations(self,int newNumMCTSSims):
		self.numMCTSSims=newNumMCTSSims

	def increaseNumSimulations(self):
		self.numMCTSSims+=1

	def exploitabilitySearch(self,localGame,belief,prevGame,prevAction, nextP=1.): #it returns net winnings for exploiting strategy by doing an exhaustive search

			if prevGame.getPlayer()!=self.game.getPlayer(): #if the previous player was the exploited one, update the belief # Otimize -E
					belief=np.multiply(belief,nextP)
					if np.sum(belief)!=0:
						belief=np.divide(belief,np.sum(belief))

			if not (prevGame.cards[2]== localGame.cards[2]): #time to set the public card (or something)
				Ppub=np.sum(belief,axis=1) #marginalize over the opponent card
				Qsc=np.zeros((3,1)) #numcards,1
				for pubCard in range(3):
					nextGame=prevGame.copy() #copy the current game
					nextGame.manualPublicCard=pubCard;
					nextGame.action(prevAction) #put a different public card on it and repeat the move
					updatedBelief=np.multiply(belief.T,nextGame.publicCardArray).T
					if np.sum(updatedBelief)!=0:
						updatedBelief=updatedBelief/np.sum(updatedBelief)
					Qsc[pubCard]=self.exploitabilitySearch(nextGame,belief=updatedBelief,nextP=nextP,prevGame=nextGame,prevAction=prevAction)
				return np.dot(Ppub,Qsc)
			else:

				exploitingPlayer= (localGame.getPlayer()==self.game.getPlayer())

				#if state is terminal return the value
				if localGame.isFinished(): # check if s is a known terminal state
					Vs=0.
					for oppCard in range(3):
						nextGame=prevGame.copy()
						if nextGame.getPlayer()==self.game.getPlayer():
							nextGame.setOpponentCard(oppCard)
						else:
							nextGame.setPlayerCard(oppCard)
						nextGame.action(action=prevAction)
						vTerm=nextGame.getOutcome()[self.game.getPlayer()]
						Vs+=vTerm*(np.sum(belief,axis=0)[oppCard])
					return Vs #Always get outcome for exploiting player

				#initialize values and keep track of bets or something
				numActions=localGame.params["actionSize"]
				bets=np.zeros((numActions,1))
				Qsa=np.zeros((numActions,1))

				if not exploitingPlayer:  #if the strategy player plays return p.v (fut)
					Ps=np.zeros((3,numActions)) #Explicit reference to number of cards in leduc
					for oppCard in range(3): #explicit reference to number of cards in leduc
						localGame.setPlayerCard(oppCard)
		
						Ps[oppCard],_ = self.nnets.policyValue(localGame.getPlayerCard(), localGame.getPublicHistory(), localGame.getPublicCard()) #get probabilities for each card, do not browse the dict, it is slower for some reason -E
						#s = localGame.playerInfoStringRepresentation() #We cannot use the information because it has been raised to the Temp
						#if s not in self.Ps: #
						#Ps[oppCard]=self.Ps[s]
						#Ps[oppCard,:]=[0.5,0.5,0.]
					Pa=np.dot(np.sum(belief,axis=0),Ps) #sum over public cards, axis 0

					#probability of taking an action
					for a in range(numActions): #Make copies of the game with each action
						nextGame=localGame.copy()
						bets[a]=nextGame.action(action=a)
						if Pa[a] !=0:
							Qsa[a]=self.exploitabilitySearch(nextGame,belief=belief,nextP=Ps[:,a],prevGame=localGame,prevAction=a)#or something like that
					return np.dot(Pa,Qsa)

				else:#if the exploiting player plays return max(v), they only take optimal actions. this also takes into account the bet made at the time
					for a in range(numActions):
						nextGame=localGame.copy()
						bets[a]=nextGame.action(action=a)
						Qsa[a]=self.exploitabilitySearch(nextGame,belief,prevGame=localGame,prevAction=a)

					Vs=Qsa-bets
					return max(Vs)

	def findAnalyticalExploitability(self):

		start = time.time()
		self.game = LeducGame() #explicit reference to leduc
		numPlayers=2 #2 player game
		numCards=3 #Jack, Queen, King
		cardCopies=2
		numActions=self.game.params["actionSize"]
		exploitability=-1.
		exploitingPlayerId=1; #Id of the player that is exploited
		self.game.resetGame()
		self.game.setPlayer(exploitingPlayerId)
		bets=np.zeros((numActions,1),dtype=float)
		V_pfirst=0.
		V_ofirst=0.
		self.gameCopy=self.game.copy()#copy a fresh game
		for firstToPlay in range(2): #two players
			#Say the exploiting player starts:
			for pCard in range(numCards): #take a card
				Qsa=np.zeros((numActions,1),dtype=float)
				self.gameCopy.setPlayerCard(pCard) #set the card specified by for loop
				belief=np.zeros((numCards,numCards),dtype=float) #this is the joint (conditional) probability for opponent card and public card
				for oppCard in range(numCards):
					for pubCard in range(numCards):
						belief[pubCard,oppCard]=1.*(2-(oppCard==pCard))*(2-(pubCard==oppCard)-(pubCard==pCard))/60
				belief=belief/np.sum(belief)
				if firstToPlay==exploitingPlayerId: #if the exploiting player starts, play the 
					for a in range(numActions):
						self.gameCopy.resetGame()
						self.gameCopy.setPlayer(firstToPlay)
						self.gameCopy.dealer=firstToPlay #This is super important for consistent results
						self.gameCopy.setPlayerCard(pCard)
						nextGame=self.gameCopy.copy()
						bets[a]=nextGame.action(action=a)	#copy game and take action
						Qsa[a]=self.exploitabilitySearch(nextGame,belief=belief,prevGame=self.gameCopy,prevAction=a)
					Vs=Qsa-bets
					V_pfirst+=(1./3)*max(Vs)
				else: #The other player starts
					self.gameCopy.resetGame()
					self.gameCopy.setPlayer(firstToPlay)
					self.gameCopy.dealer=firstToPlay
					self.gameCopy.setOpponentCard(pCard)
					nextGame=self.gameCopy.copy()
					Vs=self.exploitabilitySearch(nextGame,belief=belief,prevGame=self.gameCopy,prevAction=-1)
					V_ofirst+=(1./3)*Vs
		exploitability+=0.5*V_ofirst+0.5*V_pfirst
		#Say the exploited player starts

		end = time.time()
		print("Exploitability calculation time: "+str(end - start))
		return exploitability




