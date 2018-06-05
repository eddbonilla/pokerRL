cimport cython
import numpy as np
cimport numpy as np
from game_c cimport Game
import array
import random

cdef class HoldEmGame(Game):

	#2 = K, 1 = Q, 0 = J
	def getBlinds(self):
		return [1,2]


	def __init__(self, int override = False, int player = 0, int pot = 0, int round = 0, int bet = 0, int raisesInRound = 0, int finished = 0, np.ndarray playerCards = None, np.ndarray winnings = None, np.ndarray playersCardsArray = None, np.ndarray publicCardArray=None, np.ndarray history = None):
		if override:
			self.player = player

			self.pot = pot
			self.round = round
			self.bet = bet
			self.raisesInRound = raisesInRound
			self.finished= finished
			self.playerCards = playerCards
			self.winnings= winnings
			self.playersCardsArray = playersCardsArray
			self.publicCardArray = publicCardArray
			self.publicCardArray_view = self.publicCardArray
			self.history = history

			self.playerCards_view = self.playerCards
			self.winnings_view = self.winnings
			self.history_view = self.history

		else:
			self.resetGame()


	cpdef void resetGame(self):
		self.player = 0  #0 for player 1, 1 for player 2
		self.pot = 3 
		self.playerCards = -np.ones((2,2),dtype = int)
		self.publicCards = -np.ones(5,dtype=int)
		self.deck = np.ones(52)
		self.playersCardsArray = np.zeros((2,52))
		self.publicCardArray = np.zeros(52, dtype =int)

		for player in range(2):
			for i in range(2):
				self.playerCards[player,i] = np.random.choice(52, p = self.deck/np.sum(self.deck))
				self.deck[self.playerCards[player,i]] = 0 
				self.playersCardsArray[player][self.playerCards[player,i]]

		self.round = 0   #0 for 1st round, 1 for 2nd round
		self.bet = 2
		self.finished = False
		self.raisesInRound = 0
		self.history = np.zeros((2,4,5,2), dtype = int)
		self.history_view = self.history 
		self.winnings = np.zeros(2, dtype = int) 

	cpdef object copy(self):
		return HoldEmGame(override=True,player = self.player, pot = self.pot, round = self.round, bet = self.bet, raisesInRound = self.raisesInRound, finished = self.finished, playerCards = np.copy(self.playerCards), publicCards = self.publicCards winnings = np.copy(self.winnings), playersCardsArray = np.copy(self.playersCardsArray), publicCardsArray=np.copy(self.publicCardsArray), history = np.copy(self.history))



	cdef str playerInfoStringRepresentation(self):
		cdef str string =  str(self.player)+str(self.playerCards[self.player])+str(self.history)+str(self.publicCards)
		return string

	cdef str publicInfoStringRepresentation(self):
		cdef str string = str(self.player)+str(self.history)+str(self.publicCards)
		return string

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void finishGame(self,int playerfolded = -1):
		self.finished = 1
		cdef list value
		if playerfolded >= 0:
			self.winnings_view[(1+playerfolded)%2] = self.pot
		else:
			for i in range(2):
				hand[i] = Evaluator.evaluate(playerCards[i],publicCards)
			if hand[0]<hand[1]:	
				self.winnings[0] = self.pot
			elif 	hand[0]>hand[1]:
				self.winnings[1] = self.pot
			else:
				self.winnings[0] = self.pot // 2
				self.winnings[1] = self.pot // 2
		#print(self.winnings)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void endRound(self):
		if self.round==3:
			self.finishGame
		elif self.round==0:
			for i in range(3):
				self.publicCards[i] = np.random.choice(52, p = self.deck/np.sum(self.deck))
				self.deck[self.publicCards[i]] = 0 
				self.publicCardsArray[self.publicCards[i]]
		elif self.round == 1:
			self.publicCards[3] = np.random.choice(52, p = self.deck/np.sum(self.deck))
			self.deck[self.publicCards[3]] = 0 
			self.publicCardsArray[self.publicCards[i]]
			self.bet = 4
		elif self.round==2:
			self.publicCards[3] = np.random.choice(52, p = self.deck/np.sum(self.deck))
			self.deck[self.publicCards[3]] = 0 
			self.publicCardsArray[self.publicCards[i]]

		self.round += 1
		self.raisesInRound = 0
		self.player = 0

	cpdef int getPlayer(self):
		return self.player

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef np.ndarray getPlayerCard(self):
		return self.playersCardsArray[self.player]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef np.ndarray getOpponentCard(self):
		return self.playersCardsArray[(self.player + 1) % 2]

	cpdef np.ndarray getPlayerStates(self):
		return self.playersCardsArray

	cpdef np.ndarray getPublicCard(self):
		return self.publicCardArray
		
	cpdef int getPot(self):
		return self.pot

	cpdef int getRound(self):
		return self.round

	cdef void setPlayer(self,int player):
			self.player = player

	@cython.boundscheck(False)
	cpdef np.ndarray getPublicHistory(self):
		#Public history returned with player history at top
		cdef np.ndarray public_history = self.history[::(-1)**self.player,:,:,:]
		return public_history

	cpdef np.ndarray getOutcome(self):
		#Returns (player1reward,player2reward). NOTE reward is the pot (if game won) or zero the costs of the bets over the course of the game
		#are NOT subtracted
		return self.winnings
				
	cpdef int isFinished(self):
		return self.finished

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef np.ndarray regulariseOpponentEstimate(self,np.ndarray estimate):
		cdef np.ndarray mask = 1 - (0.5+0.5*(self.cards[self.player] == self.cards[2]))*self.playersCardsArray[self.player]
		cdef np.ndarray probs= mask*estimate
		probs /= np.sum(probs)
		return probs

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef int action(self,int action=-1, np.ndarray strategy=None):
		#Randomly select action from strategy.
		#Args:
		#	strategy (list(float)): Strategy of the node
		#Returns:
		#	Chosen action, bet size
		if self.finished:
			return -1,-1
		if action==-1:
			action = np.random.choice(3, p = strategy)
		cdef int oldPlayer = self.player
		cdef int oldRaisesInRound = self.raisesInRound
		cdef int oldRound = self.round
		self.player = (oldPlayer + 1)%2
		cdef int endRound = 0
		cdef int betAmount = 0
		#print("Action")
		if action ==2:
			self.finishGame(oldPlayer)
			#print("Fold")
		else:
			if oldRound == 0 and oldRaisesInRound == 0 and oldPlayer == 0:
				betAmount+=1
			if oldRaisesInRound == 4:
				betAmount += self.bet
				self.history_view[oldPlayer,oldRound,oldRaisesInRound,1] = 1
				endRound = 1
			#print("Two raises + call/raise")
			elif action == 1:
				if oldRaisesInRound>=1:
					betAmount += self.bet
					endRound = 1
				#print("One raise + call")
				elif oldRaisesInRound == 0 and (oldPlayer == 1):
					endRound = 1				
					#print("Call + end round")
				#else:
				#print("Call")
				self.history_view[oldPlayer,oldRound,oldRaisesInRound,1] = 1
			elif action == 0:
				if oldRaisesInRound>=1:
					betAmount += 2*self.bet
				#print("Second Raise")
				else:
					betAmount += self.bet
				#print("First raise")
				self.history_view[oldPlayer,oldRound,oldRaisesInRound,0] = 1
				self.raisesInRound += 1
		self.pot += betAmount
		if endRound:
			self.endRound()
		return betAmount



