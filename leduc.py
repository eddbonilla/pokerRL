import numpy as np
import random

class Game:
	pass

class LeducGame(Game):
	#2 = K, 1 = Q, 0 = J
	def getAnte(self):
		return 1

	params = {"inputSize" : 30, "historySize" : 24, "handSize" : 3, "publicCardSize" : 3, "actionSize" : 3, "valueSize": 1}


	def __init__(self):
		self.dealer = random.randint(0,1)
		self.player = self.dealer  #0 for player 1, 1 for player 2
		self.pot = 2
		self.cards = {}
		self.cards["player1"] = random.randint(0,2)
		self.cards["player2"] = (self.cards["player1"] + (random.randint(0,4)%3) + 1)%3
		self.round = 0   #0 for 1st round, 1 for 2nd round
		self.bet = 2
		self.finished = False
		self.playerfolded = None
		self.raisesInRound = 0
		self.history = np.zeros((2,2,3,2))
		self.winnings = None

	def resetGame(self):
		self.dealer = random.randint(0,1)
		self.player = self.dealer  #0 for player 1, 1 for player 2
		self.pot = 2
		if "public" in self.cards: 
			del self.cards["public"] 
		self.cards["player1"] = random.randint(0,2)
		self.cards["player2"] = (self.cards["player1"] + (random.randint(0,4)%3) + 1)%3
		self.round = 0   #0 for 1st round, 1 for 2nd round
		self.bet = 2
		self.finished = False
		self.playerfolded = None
		self.raisesInRound = 0
		self.history = np.zeros((2,2,3,2))
		self.winnings = None


	def playerInfoStringRepresentation(self):
		dict = { 	"player" : self.player,
					"playerCard": self.cards["player" + str(self.player+1)], 
					"publicHistory": self.getPublicHistory()}
		if "public" in self.cards:
			dict["publicCard"] = self.cards["public"]

		return str(dict)

	def publicInfoStringRepresentation(self):
		dict = { 	"player" : self.player, 
					"publicHistory": self.getPublicHistory()}
		if "public" in self.cards:
			dict["publicCard"] = self.cards["public"]

		return str(dict)

	def finishGame(self,playerfolded):
		self.finished = True
		self.playerfolded = playerfolded
		self.winnings = np.zeros(2)
		if not playerfolded == None:
			self.winnings[(1+playerfolded)%2] = self.pot
		elif self.cards["player1"] == self.cards["public"]:
			self.winnings[0] = self.pot
		elif self.cards["player2"] == self.cards["public"]:
			self.winnings[1] = self.pot
		elif self.cards["player1"] > self.cards["player2"] and self.cards["player1"] > self.cards["public"]:
			self.winnings[0] = self.pot
		elif self.cards["player2"] > self.cards["player1"] and self.cards["player2"] > self.cards["public"]:
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
			if self.cards["player1"] == self.cards["player2"]:
				self.cards["public"] = (self.cards["player1"] + 1 + random.randint(0,1))%3
			else:
				self.cards["public"] = (random.randint(0,3) - self.cards["player1"] - self.cards["player2"]) % 3
		else:
			self.finishGame(None)

	def getPlayer(self):
		return self.player

	def getPlayerCard(self):
		return np.eye(3)[self.cards["player"+str(self.player+1)]]

	def getOpponentCard(self):
		return np.eye(3)[self.cards["player"+str(2 - self.player)]]

	def getPlayerStates(self):
		#return tf.one_hot(self.cards["player1"],3),tf.one_hot(self.cards["player2"],3)
		return np.eye(3)[[self.cards["player1"],self.cards["player2"]]]

	def getPublicCard(self):
		if "public" in self.cards:
			publicCard = (np.eye(3)[[self.cards["public"]]]).flatten()
		else:
			publicCard = np.zeros(3)
		return publicCard
		
	def getPot(self):
		return self.pot

	def setOpponentCard(self,card):
		#input: card as scalar number e.g. 2=K,1=Q,0=J
		self.cards["player"+str(2 - self.player)] = card 

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

	def action(self,strategy):
		#Randomly select action from strategy.
		#Args:
		#	strategy (list(float)): Strategy of the node
		#Returns:
		#	Chosen action, bet size
		if self.finished:
			return None
		choice = random.random()
		action = 2
		betAmount = 0
		probability_sum = 0
		probabilities=[]
		for i in range(3):
			action_probability = strategy[i]
			if action_probability == 0:
				continue
			probability_sum += action_probability
			if choice < probability_sum:
				action = i
				break
		oldPlayer = self.player
		oldRaisesInRound = self.raisesInRound
		oldRound = self.round
		self.player = (oldPlayer + 1)%2
		endRound = False
		if action ==2:
			self.finishGame(oldPlayer)
		elif oldRaisesInRound == 2:
			betAmount = self.bet
			self.history[oldPlayer,oldRound,oldRaisesInRound,1] = 1
			endRound = True
		elif action == 1:
			if oldRaisesInRound==1:
				betAmount = self.bet
				endRound = True
			elif oldRaisesInRound == 0 and oldPlayer == 1:
				endRound = True				
			self.history[oldPlayer,oldRound,oldRaisesInRound,1] = 1
		if action == 0:
			if oldRaisesInRound==1:
				betAmount = 2*self.bet
			else:
				betAmount = self.bet
			self.history[oldPlayer,oldRound,oldRaisesInRound,0] = 1
			self.raisesInRound += 1
		self.pot += betAmount
		if endRound:
			self.endRound()
		return action,betAmount



