cimport cython
import numpy as np
cimport numpy as np
import array
import random

cdef class LeducGame():

	#2 = K, 1 = Q, 0 = J
	def getAnte(self):
		return 1


	def __init__(self, int override = False, int player = 0, int pot = 0, int round = 0, int bet = 0, int raisesInRound = 0, int finished = 0, np.ndarray cards = None, np.ndarray winnings = None, np.ndarray playersCardsArray = None, np.ndarray publicCardArray=None, np.ndarray history = None):
		if override:
			self.player = player

			self.pot = pot
			self.round = round
			self.bet = bet
			self.raisesInRound = raisesInRound
			self.finished= finished
			self.cards = cards
			self.winnings= winnings
			self.playersCardsArray = playersCardsArray
			self.publicCardArray = publicCardArray
			self.publicCardArray_view = self.publicCardArray
			self.history = history

			self.cards_view = self.cards
			self.winnings_view = self.winnings
			self.history_view = self.history

		else:
			self.resetGame()


	cpdef void resetGame(self):
		self.player = 0  #0 for player 1, 1 for player 2
		self.pot = 2 
		self.cards = np.zeros(3,dtype = int)
		self.cards_view = self.cards
		self.cards_view[0] = random.randint(0,2)
		self.cards_view[1] = (self.cards[0] + (random.randint(0,4)%3) + 1)%3
		self.cards_view[2] = -1
		self.playersCardsArray = np.eye(3, dtype = int)[self.cards_view[0:2]]
		self.publicCardArray = np.zeros(3, dtype =int)
		self.publicCardArray_view = self.publicCardArray
		self.round = 0   #0 for 1st round, 1 for 2nd round
		self.bet = 2
		self.finished = False
		self.raisesInRound = 0
		self.history = np.zeros((2,2,3,2), dtype = int)
		self.history_view = self.history 
		self.winnings = np.zeros(2, dtype = int) 
		self.winnings_view = self.winnings

	cpdef object copy(self):
		cdef object newGame = LeducGame(override=True,player = self.player, pot = self.pot, round = self.round, bet = self.bet, raisesInRound = self.raisesInRound, finished = self.finished, cards = np.copy(self.cards), winnings = np.copy(self.winnings), playersCardsArray = np.copy(self.playersCardsArray), publicCardArray=np.copy(self.publicCardArray), history = np.copy(self.history))

		return newGame


	cdef str playerInfoStringRepresentation(self):
		cdef str string =  str(self.player)+str(self.cards[self.player])+str(self.history)+str(self.getPublicCard())
		return string

	cdef str publicInfoStringRepresentation(self):
		cdef str string = str(self.player)+str(self.history)+str(self.getPublicCard())
		return string

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void finishGame(self,int playerfolded = -1):
		self.finished = 1
		if playerfolded >= 0:
			self.winnings_view[(1+playerfolded)%2] = self.pot
		elif self.cards_view[0] == self.cards_view[2]:
			self.winnings_view[0] = self.pot
		elif self.cards_view[1] == self.cards_view[2]:
			self.winnings_view[1] = self.pot
		elif self.cards_view[0] > self.cards_view[1] and self.cards_view[0] > self.cards_view[2]:
			self.winnings_view[0] = self.pot
		elif self.cards_view[1] > self.cards_view[0] and self.cards_view[1] > self.cards_view[2]:
			self.winnings_view[1] = self.pot
		else:
			self.winnings += self.pot/2
		#print(self.winnings)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void endRound(self):
		if self.round==0:
			self.round = 1
			self.bet = 4
			self.raisesInRound = 0
			self.player = 0
			if self.cards_view[0] == self.cards_view[1]:
				self.cards_view[2] = (self.cards_view[0] + 1 + random.randint(0,1))%3
			else:
				self.cards_view[2] = (random.randint(0,3) - self.cards_view[0] - self.cards_view[1]) % 3
			self.publicCardArray_view[self.cards_view[2]] = 1
		else:
			self.finishGame()

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

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef void setOpponentCard(self,int card):
		#input: card as scalar number e.g. 2=K,1=Q,0=J
		self.cards_view[(self.player+1)%2] = card 
		self.playersCardsArray[(self.player + 1) % 2] = np.eye(3,dtype = int)[card]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef void setPlayerCard(self,int card):
		#input: card as scalar number e.g. 2=K,1=Q,0=J
		self.cards_view[self.player] = card 
		self.playersCardsArray[self.player] = np.eye(3, dtype = int)[card]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cpdef void setPublicCard(self,int card):
		#input: card as scalar number e.g. 2=K,1=Q,0=J
		self.cards_view[2] = card 
		self.publicCardArray = np.eye(3, dtype = int)[card]
		self.publicCardArray_view = self.publicCardArray

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
		elif oldRaisesInRound == 2:
			betAmount = self.bet
			self.history_view[oldPlayer,oldRound,oldRaisesInRound,1] = 1
			endRound = 1
			#print("Two raises + call/raise")
		elif action == 1:
			if oldRaisesInRound==1:
				betAmount = self.bet
				endRound = 1
				#print("One raise + call")
			elif oldRaisesInRound == 0 and (oldPlayer == 1):
				endRound = 1				
				#print("Call + end round")
			#else:
				#print("Call")
			self.history_view[oldPlayer,oldRound,oldRaisesInRound,1] = 1
		elif action == 0:
			if oldRaisesInRound==1:
				betAmount = 2*self.bet
				#print("Second Raise")
			else:
				betAmount = self.bet
				#print("First raise")
			self.history_view[oldPlayer,oldRound,oldRaisesInRound,0] = 1
			self.raisesInRound += 1
		self.pot += betAmount
		if endRound:
			self.endRound()
		return betAmount



