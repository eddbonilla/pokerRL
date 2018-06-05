#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)
#DEPRECATED USE MCTS_c
import sys
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

	def __init__(self, nnets, numMCTSSims, cpuct, temp = 2, floor = 0.05):
		self.game = None
		self.gameCopy= None;
		self.nnets = nnets #neural networks used to predict the cards and/or action probabilities
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct
		self.temp = temp
		self.floor = floor	#Creates a floor p value to ensure all possible paths are explored
		self.Qsa = {}	   # stores Q values for s,a (as defined in the paper)
		self.Nsa = {}	   # stores #times edge s,a was visited
		self.Ns = {}		# stores #times board s was visited
		self.Ps = {}		# stores initial policy (returned by neural net)


		self.Es = {}		# stores game.getGameEnded ended for board s
		self.Bs = {} 		# stores best response to avg strategies for board s
		#self.Vs = {}		# stores game.getValidMoves for board s

		#Harcoded Parameters. To be deprecated soon
		self.tempDecayRate = 1.002
		self.addSims= 5
		self.minTemp= 0.1

	def reduceTemp(self):

		if self.temp > self.minTemp: #hardcoded final temperature
			self.temp = self.temp/self.tempDecayRate

	def increaseNumSimulations(self):
		self.numMCTSSims+=self.addSims

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
		print(self.temp)
		self.game=game
		estimOpponentCards= self.game.regulariseOpponentEstimate(self.nnets.estimateOpponent(self.game.getPublicHistory(), self.game.getPublicCard())) # gives a guess of the opponent cards, we can change this to be the actual cards
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
		#averageStrategy = self.Ps[s]
		return treeStrategy 		#return pi,tree strategy

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
				self.Ps[s] = self.Ps[s]**(self.temp) #What? 
				self.Qsa[s] = v * np.ones(self.game.params["actionSize"])
				self.Ns[s] = 0
				self.Nsa[s] = np.zeros(self.game.params["actionSize"])
				return (v*pot)
				#if exploitSearch:
				#	self.Ps[s] = np.ones(3)/3
			elif s_pub not in self.Ns:
				self.Ns[s_pub] = 0
				self.Nsa[s_pub] = np.zeros(self.game.params["actionSize"])

		# pick the action with the highest upper confidence bound

		if playerMove:
			u = self.Qsa[s] + math.sqrt(self.Ns[s]+EPS)*self.cpuct*(self.Ps[s]+self.floor)/(1+self.Nsa[s])

		else:
			u = self.cpuct*self.Ps[s]*math.sqrt(self.Ns[s_pub]+EPS)/(1+self.Nsa[s_pub])

		#print(u)
		a=np.argmax(u)
		#print("probs =" +str(self.Ps[s])+", playerMove = "+str(playerMove)+ ", action ="+str(a))

		bet = self.gameCopy.action(action=a)
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
		
	def setTreeSearchParams(self,params):
		self.numMCTSSims=params["numMCTS"]
		self.tempDecayRate=params["tempDecayRate"]

	def exploitabilitySearch(self,localGame,belief,prevGame,prevAction): #it returns net winnings for exploiting strategy by doing an exhaustive search
		
		if not (prevGame.cards[2]== localGame.cards[2]): #time to set the public card (or something)
			Ppub=np.sum(belief,axis=2) ##for pcard inside the mix
			Qsc=np.zeros((3,3)) #numcards,1
			for pubCard in range(3):

				nextGame=copy.deepcopy(localGame) #copy the current game
				nextGame.setPublicCard(pubCard)
				updatedBelief=np.transpose(np.multiply(np.transpose(belief,axes=[0,2,1]),nextGame.publicCardArray),axes=[0,2,1])
				summ=np.sum(updatedBelief,axis=(1,2))

				for pCard in range(3):
					if summ[pCard]!=0:
						updatedBelief[pCard,:,:]=updatedBelief[pCard,:,:]/summ[pCard]

				Qsc[:,pubCard]=np.reshape(self.exploitabilitySearch(nextGame,belief=updatedBelief,prevGame=nextGame,prevAction=prevAction),3)
			
			return np.diag(np.dot(Qsc,Ppub.T))

		else:

			#if state is terminal return the value
			if localGame.isFinished(): # check if s is a known terminal state
				Vs=np.zeros((3,1),dtype=float) #explicit reference to number of cards
				for pCard in range(3):
					for oppCard in range(3):
						nextGame=copy.deepcopy(prevGame)
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
			exploitingPlayer= (localGame.getPlayer()==self.game.getPlayer())
			numActions=localGame.params["actionSize"]
			Qsa=np.zeros((3,numActions)) #explicit reference to the number of cards


			if not exploitingPlayer:  #if the strategy player plays return p.v (fut)
				Ps=np.zeros((3,numActions)) #Explicit reference to number of cards in leduc
				for oppCard in range(3): #explicit reference to number of cards in leduc
					localGame.setPlayerCard(oppCard)
	
					#Ps[oppCard,:],_ = self.nnets.policyValue(localGame.getPlayerCard(), localGame.getPublicHistory(), localGame.getPublicCard()) #get probabilities for each card, do not browse the dict, it is slower for some reason -E
					#s = localGame.playerInfoStringRepresentation() #We cannot use the information because it has been raised to the Temp
					#if s not in self.Ps: #
					#Ps[oppCard]=self.Ps[s]
					Ps[oppCard,:]=[0.,1.,0.]
				Pa=np.dot(np.sum(belief,axis=1),Ps) #marginalize over public cards, axis 1. Probability of taking an action a 
				for a in range(numActions): #Make copies of the game with each action
					if Pa[:,a].any() !=0: #if the action has any probability of happening
						nextGame=copy.deepcopy(localGame)
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
				bets=np.zeros((1,numActions))
				cardV=np.zeros(3)
				for a in range(numActions):
					nextGame=copy.deepcopy(localGame)
					bets[:,a]=nextGame.action(action=a)
					Qsa[:,a]=np.reshape(self.exploitabilitySearch(nextGame,belief,prevGame=localGame,prevAction=a),3)
				Vs=Qsa-bets
				self.Bs[localGame.publicInfoStringRepresentation()]=np.argmax(Vs,axis=1) #store the optimal action in the dict, index is the player card
				return  np.max(Vs,axis=1)#take the max over actions of Vs

	def findAnalyticalExploitability(self):

		start = time.time()
		self.game = LeducGame() #explicit reference to leduc
		numPlayers=2 #2 player game
		numCards=3 #Jack, Queen, King
		cardCopies=2
		numActions=self.game.params["actionSize"]
		exploitability=-1.
		exploitingPlayerId=1; #Id of the player that is exploited

		self.gameCopy=copy.deepcopy(self.game)#copy a fresh game

		#create an array with the conditional probabilities of being in a state
		belief=np.zeros((numCards,numCards,numCards),dtype=float) #exploiting player card, public card,exploited player card
		for pCard in range(numCards): #take a card
			Qsa=np.zeros((numActions,1),dtype=float)
			self.gameCopy.setPlayerCard(pCard) #set the card specified by for loop
			for oppCard in range(numCards):
				for pubCard in range(numCards):
					belief[pCard,pubCard,oppCard]=1.*(2-(oppCard==pCard))*(2-(pubCard==oppCard)-(pubCard==pCard))/20.

		for firstToPlay in range(2): #two players
			self.game.setPlayer(firstToPlay) #set the exploiting player to start both first and second

			nextGame=copy.deepcopy(self.gameCopy)

			#start the hunt
			Vs=self.exploitabilitySearch(nextGame,belief=belief,prevGame=self.gameCopy,prevAction=-1)
			exploitability+=(1./3)*(np.sum(Vs)/2) #3 cards, two starting positions

		end = time.time()
		print("Exploitability calculation time: "+str(end - start))
		return exploitability, self.Bs

