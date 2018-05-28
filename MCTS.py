#Code adapted from https://github.com/suragnair/alpha-zero-general (needs heavy adaptation for imperfect information games)
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

	def __init__(self, nnets, numMCTSSims, cpuct, temp = 1, floor = 0.05):
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
		#self.Vs = {}		# stores game.getValidMoves for board s

		#Harcoded Parameters. To be deprecated soon
		self.tempDecayRate = 1.002

	def reduceTemp(self):

		if self.temp > 0.5: #hardcoded final temperature
			self.temp = self.temp/self.tempDecayRate



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
		exploitability = -1.
		for card in range(3):
			history = np.zeros((2,2,3,2))
			publicCard = np.zeros(3)
			print(self.Qsa['0'+str(card)+str(history)+str(publicCard)])
			exploitability += 1./3*np.max(self.Qsa['0'+str(card)+str(history)+str(publicCard)])
			for oppCard in range(3):
				p = self.Ps['1'+str(oppCard)+str(history)+str(publicCard)]
				history[1,0,0,0] = 1
				raiseString = '0'+str(card)+str(history)+str(publicCard)
				history[1,0,0,0] = 0
				history[1,0,0,1] = 1
				checkString = '0'+str(card)+str(history)+ str(publicCard)
				history[1,0,0,1] = 0
				if raiseString in self.Qsa:
					exploitability += 1./9*p[0]*np.max(self.Qsa[raiseString])
				if checkString in self.Qsa:
					exploitability +=1./9*p[1]*np.max(self.Qsa[checkString])
					exploitability +=1./9*p[2]
		return exploitability
		
	def setTreeSearchParams(self,params):
		self.numMCTSSims=params["initialNumMCTS"]
		self.temp=params["initialTreeTemperature"]
		self.tempDecayRate=params["tempDecayRate"]
		self.cpuct=params["cpuct"]

	def exploitabilitySearch(self,localGame,belief,prevGame,prevAction): #it returns net winnings for exploiting strategy by doing an exhaustive search
		if not (prevGame.cards[2]== localGame.cards[2]): #time to set the public card (or something)
			#Ppub=np.sum(belief,axis=1) #marginalize over the opponent card ##
			Ppub=np.sum(belief,axis=2) ##for pcard inside the mix
			Qsc=np.zeros((3,3)) #numcards,1
			for pubCard in range(3):
				nextGame=copy.deepcopy(prevGame) #copy the current game
				nextGame.manualPublicCard=pubCard;
				nextGame.action(prevAction) #put a different public card on it and repeat the move
				updatedBelief=np.transpose(np.multiply(np.transpose(belief,axes=[0,2,1]),nextGame.publicCardArray),axes=[0,2,1])
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
			numActions=localGame.params["actionSize"]
			Qsa=np.zeros((3,numActions)) #explicit reference to the number of cards

			if not exploitingPlayer:  #if the strategy player plays return p.v (fut)
				Ps=np.zeros((3,numActions)) #Explicit reference to number of cards in leduc
				for oppCard in range(3): #explicit reference to number of cards in leduc
					localGame.setPlayerCard(oppCard)
	
					Ps[oppCard,:],_ = self.nnets.policyValue(localGame.getPlayerCard(), localGame.getPublicHistory(), localGame.getPublicCard()) #get probabilities for each card, do not browse the dict, it is slower for some reason -E
					#s = localGame.playerInfoStringRepresentation() #We cannot use the information because it has been raised to the Temp
					#if s not in self.Ps: #
					#Ps[oppCard]=self.Ps[s]
					#Ps[oppCard,:]=[0.5,0.5,0.]
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
				return  np.maximum(np.maximum(Vs[:,0],Vs[:,1]),Vs[:,2])#take the max over actions of Vs

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

		self.gameCopy=copy.deepcopy(self.game)#copy a fresh game

		#create an array with the conditional probabilities of being in a state
		belief=np.zeros((numCards,numCards,numCards),dtype=float)
		for pCard in range(numCards): #take a card
			Qsa=np.zeros((numActions,1),dtype=float)
			self.gameCopy.setPlayerCard(pCard) #set the card specified by for loop
			for oppCard in range(numCards):
				for pubCard in range(numCards):
					belief[pCard,pubCard,oppCard]=1.*(2-(oppCard==pCard))*(2-(pubCard==oppCard)-(pubCard==pCard))/20.

		for firstToPlay in range(2): #two players
			self.gameCopy.setPlayer(firstToPlay)
			self.gameCopy.dealer=firstToPlay #This is super important for consistent results

			#Say the exploiting player starts:
			if firstToPlay==exploitingPlayerId: #if the exploiting player starts, play the 
				self.gameCopy.setPlayerCard(pCard)
				nextGame=copy.deepcopy(self.gameCopy)

			else: #The other player starts
				self.gameCopy.setOpponentCard(pCard)
				nextGame=copy.deepcopy(self.gameCopy)

			#start the hunt
			Vs=self.exploitabilitySearch(nextGame,belief=belief,prevGame=self.gameCopy,prevAction=-1)
			exploitability+=(1./3)*(np.sum(Vs)/2) #3 cards, two starting positions

		end = time.time()
		print("Exploitability calculation time: "+str(end - start))
		return exploitability

