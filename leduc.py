import numpy as np
import random

class Game:
	pass

class LeducGame(Game):
	#2 = K, 1 = Q, 0 = J
	def getAnte(self):
		return 1

	params = {"inputSize" : 30, "historySize" : 24, "handSize" : 3, "actionSize" : 3, "valueSize": 1}


	def __init__(self):
		self.resetGame()

	def resetGame(self):
		self.dealer = random.randint(0,1)
		self.player = self.dealer  #0 for player 1, 1 for player 2
		self.pot = 2
		self.cards = np.zeros(3,dtype = int) 
		self.cards[0] = random.randint(0,2)
		self.cards[1] = (self.cards[0] + (random.randint(0,4)%3) + 1)%3
		self.cards[2] = -1
		self.playersCardsArray = np.eye(3)[self.cards[0:2]]
		self.publicCardArray = np.zeros(3)
		self.round = 0   #0 for 1st round, 1 for 2nd round
		self.bet = 2
		self.finished = False
		self.playerfolded = None
		self.raisesInRound = 0
		self.history = np.zeros((2,2,3,2))
		self.winnings = None


	def playerInfoStringRepresentation(self):
		return (str(self.player)+str(self.cards[self.player])+str(self.history)+str(self.getPublicCard()))

	def publicInfoStringRepresentation(self):
		return (str(self.player)+str(self.history)+str(self.getPublicCard()))

	def finishGame(self,playerfolded):
		self.finished = True
		self.playerfolded = playerfolded
		self.winnings = np.zeros(2)
		if not playerfolded == None:
			self.winnings[(1+playerfolded)%2] = self.pot
		elif self.cards[0] == self.cards[2]:
			self.winnings[0] = self.pot
		elif self.cards[1] == self.cards[2]:
			self.winnings[1] = self.pot
		elif self.cards[0] > self.cards[1] and self.cards[0] > self.cards[2]:
			self.winnings[0] = self.pot
		elif self.cards[1] > self.cards[0] and self.cards[1] > self.cards[2]:
			self.winnings[1] = self.pot
		else:
			self.winnings += self.pot/2
		#print(self.winnings)


	def endRound(self):
		if self.round==0:
			self.round = 1
			self.bet = 4
			self.raisesInRound = 0
			self.player = self.dealer
			if self.cards[0] == self.cards[1]:
				self.cards[2] = (self.cards[0] + 1 + random.randint(0,1))%3
			else:
				self.cards[2] = (random.randint(0,3) - self.cards[0] - self.cards[1]) % 3
			self.publicCardArray[self.cards[0]] = 1
		else:
			self.finishGame(None)

	def getPlayer(self):
		return self.player

	def getPlayerCard(self):
		return self.playersCardsArray[self.player]

	def getOpponentCard(self):
		return self.playersCardsArray[(self.player + 1) % 2]

	def getPlayerStates(self):
		#return tf.one_hot(self.cards["player1"],3),tf.one_hot(self.cards["player2"],3)
		return self.playersCardsArray

	def getPublicCard(self):
		return self.publicCardArray
		
	def getPot(self):
		return self.pot

	def setOpponentCard(self,card):
		#input: card as scalar number e.g. 2=K,1=Q,0=J
		self.cards[(self.player+1)%2] = card 
		self.playersCardsArray[(self.player + 1) % 2] = np.eye(3)[card]

	def setPlayer(self,player):
			self.player = player

	def getPublicHistory(self):
		#Public history returned with player history at top
		public_history = self.history[::(-1)**self.player,:,:,:]
		return public_history

	def getOutcome(self):
		#Returns (player1reward,player2reward). NOTE reward is the pot (if game won) or zero the costs of the bets over the course of the game
		#are NOT subtracted
		return self.winnings
				
	def isFinished(self):
		return self.finished

	def regulariseOpponentEstimate(self,estimate):
		mask = 1 - 0.5*self.playersCardsArray[self.player] - 0.5*self.publicCardArray
		probs= mask*estimate
		probs /= np.sum(probs)
		return probs

	def action(self,action=None,strategy=None):
		#Randomly select action from strategy.
		#Args:
		#	strategy (list(float)): Strategy of the node
		#Returns:
		#	Chosen action, bet size
		if self.finished:
			return None
		if strategy is not None:
			action = np.random.choice(3, p = strategy)
		if action is None:
			action = input("WTF?")
		oldPlayer = self.player
		oldRaisesInRound = self.raisesInRound
		oldRound = self.round
		self.player = (oldPlayer + 1)%2
		endRound = False
		betAmount = 0
		#print("Action")
		if action ==2:
			self.finishGame(oldPlayer)
			#print("Fold")
		elif oldRaisesInRound == 2:
			betAmount = self.bet
			self.history[oldPlayer,oldRound,oldRaisesInRound,1] = 1
			endRound = True
			#print("Two raises + call/raise")
		elif action == 1:
			if oldRaisesInRound==1:
				betAmount = self.bet
				endRound = True
				#print("One raise + call")
			elif oldRaisesInRound == 0 and not (oldPlayer == self.dealer):
				endRound = True				
				#print("Call + end round")
			#else:
				#print("Call")
			self.history[oldPlayer,oldRound,oldRaisesInRound,1] = 1
		elif action == 0:
			if oldRaisesInRound==1:
				betAmount = 2*self.bet
				#print("Second Raise")
			else:
				betAmount = self.bet
				#print("First raise")
			self.history[oldPlayer,oldRound,oldRaisesInRound,0] = 1
			self.raisesInRound += 1
		self.pot += betAmount
		if endRound:
			self.endRound()
		return action,betAmount



