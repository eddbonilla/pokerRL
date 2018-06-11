cimport cython
import numpy as np
cimport numpy as np
from game_c cimport Game
from evaluator import Evaluator
import array
import random

cdef class HoldEmGame(Game):

	cdef np.ndarray playerCards,publicCards, deck
	cdef int[:,:] playerCards_view
	cdef int[:] publicCards_view, deck_view
	cdef object evaluator
	#2 = K, 1 = Q, 0 = J
	def getBlinds(self):
		return np.array([1,2])


	def __init__(self, int copy = False, HoldEmGame game = None):
		if copy:
			self.player = game.player

			self.pot = game.pot
			self.round = game.round
			self.bet = game.bet
			self.raisesInRound = game.raisesInRound
			self.finished= game.finished
			self.playerCards = np.copy(game.playerCards)
			self.playerCards_view = self.playerCards
			self.publicCards = np.copy(game.publicCards)
			self.publicCards_view = self.publicCards
			self.winnings= np.copy(game.winnings)
			self.winnings_view = self.winnings
			self.playersCardsArray = np.copy(game.playersCardsArray)
			self.playerCardsArray_view = self.playersCardsArray
			self.publicCardsArray = np.copy(game.publicCardsArray)
			self.publicCardsArray_view = self.publicCardsArray
			self.history = np.copy(game.history)
			self.history_view = self.history
			self.deck = np.copy(game.deck)
			self.deck_view = self.deck

			self.evaluator = game.evaluator
			
			
			

		else:
			self.resetGame()


	cpdef void resetGame(self):
		self.player = 0  #0 for player 1, 1 for player 2
		self.pot = 3 
		
		self.playerCards = -np.ones((2,2),dtype = np.int32)
		self.playerCards_view = self.playerCards

		self.publicCards = -np.ones(5,dtype=np.int32)
		self.publicCards_view = self.publicCards

		self.deck = np.ones(52, dtype = np.int32)
		self.deck_view = self.deck

		self.playersCardsArray = np.zeros((2,52), dtype = np.int32)
		self.playerCardsArray_view = self.playersCardsArray

		self.publicCardsArray = np.zeros(52, dtype =np.int32)
		self.publicCardsArray_view = self.publicCardsArray


		for player in range(2):
			for i in range(2):
				self.playerCards_view[player,i] = np.random.choice(52, p = self.deck/np.sum(self.deck))
				self.deck_view[self.playerCards_view[player,i]] = 0 
				self.playersCardsArray[player][self.playerCards_view[player,i]] = 1

		self.round = 0   #0 for 1st round, 1 for 2nd round
		self.bet = 2
		self.finished = False
		self.raisesInRound = 0
		self.history = np.zeros((2,4,5,2), dtype = np.int32)
		self.history_view = self.history 
		self.winnings = np.zeros(2, dtype = np.int32) 
		self.winnings_view = self.winnings

		self.evaluator = Evaluator()

	cpdef object copy(self):
		return HoldEmGame(copy=True, game = self)



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
		cdef int[:] hand = np.zeros(2, dtype = np.int32)
		if playerfolded >= 0:
			self.winnings_view[(1+playerfolded)%2] = self.pot
		else:
			for i in range(2):
				hand[i] = self.evaluator.evaluate(self.playerCards_view[i],self.publicCards)
			if hand[0]<hand[1]:	
				self.winnings_view[0] = self.pot
			elif 	hand[0]>hand[1]:
				self.winnings_view[1] = self.pot
			else:
				self.winnings_view[0] = self.pot // 2
				self.winnings_view[1] = self.pot // 2
		#print(self.winnings)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void endRound(self):
		if self.round==3:
			self.finishGame()
		elif self.round==0:
			for i in range(3):
				self.publicCards_view[i] = np.random.choice(52, p = self.deck/np.sum(self.deck))
				self.deck_view[self.publicCards_view[i]] = 0 
				self.publicCardsArray_view[self.publicCards_view[i]] = 1
		elif self.round == 1:
			self.publicCards_view[3] = np.random.choice(52, p = self.deck/np.sum(self.deck))
			self.deck_view[self.publicCards_view[3]] = 0 
			self.publicCardsArray_view[self.publicCards_view[3]] = 1
			self.bet = 4
		elif self.round==2:
			self.publicCards_view[4] = np.random.choice(52, p = self.deck/np.sum(self.deck))
			self.deck_view[self.publicCards_view[4]] = 0 
			self.publicCardsArray_view[self.publicCards_view[4]] = 1

		self.round += 1
		self.raisesInRound = 0
		self.player = 0

	cpdef int getPlayer(self):
		return self.player

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef np.ndarray getPlayerCards(self):
		return self.playersCardsArray[self.player]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef np.ndarray getOpponentCards(self):
		return self.playerCards[(self.player + 1) % 2]

	cpdef np.ndarray getPlayerStates(self):
		return self.playersCardsArray

	cpdef np.ndarray getPublicCards(self):
		return self.publicCardsArray
		
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
		estimate= self.deck*estimate
		estimate /= np.sum(estimate)
		return estimate

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef int action(self,int action=-1, np.ndarray strategy=None):
		#Randomly select action from strategy.
		#Args:
		#	strategy (list(float)): Strategy of the node
		#Returns:
		#	Chosen action, bet size
		if self.finished:
			return -1
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


	cpdef int sampleOpponent(self,object nnets):
		cdef int card
		for card in self.playerCards[(self.player+1)%2]:
			self.deck_view[card] = 1
			self.playerCardsArray_view[(self.player+1)%2,card] = 0
		cdef np.ndarray probs = self.regulariseOpponentEstimate(nnets.estimateOpponent(self.getPlayerCards(), self.getPublicHistory(), self.getPublicCards()))
		cdef int firstCard = np.random.choice(52,p=probs)
		self.playerCards_view[(self.player+1)%2,0] = firstCard
		self.deck_view[firstCard] = 0
		self.playerCardsArray_view[(self.player+1)%2,firstCard] = 1
		probs = self.regulariseOpponentEstimate(nnets.estimateOpponent(self.getPlayerCards(), self.getPublicHistory(), self.getPublicCards(), firstCard = firstCard))
		cdef int secondCard = np.random.choice(52,p=probs)
		self.playerCards_view[(self.player+1)%2,1] = secondCard
		self.deck_view[secondCard] = 0
		self.playerCardsArray_view[(self.player+1)%2,secondCard] = 1

		



