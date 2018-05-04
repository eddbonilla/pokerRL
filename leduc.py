class Game:
	pass

class LeducGame(Game):
	#0 = K, 1 = Q, 2 = J

	def __init__(self):
		self.player = 1  #1 for player 1, 2 for player 2
		self.pot = 2
		self.cards["player1"] = random.randint(3)
		self.cards["player2"] = (self.cards["player1"] + (random.randint(5)%3) + 1)%3
		self.round = 1
		self.bet = 2
		self.finished = False
		self.playerfolded = None
		self.raisesInRound = 0
		self.history = 

	def endRound(self):
		if round==1:
			self.round = 2
			self.bet = 4
			self.raisesInRound = 0
			if self.cards["player1"] == self.cards["player2"]:
				self.cards["public"] = (self.cards["player1"] + 1 + random.randint(2))%3
			else:
				self.cards["public"] = (random.randint(4) - self.cards["player1"] - self.cards["player2"]) % 3
		else:
			self.finished =True

	def getPlayer():
		return self.player

	def getPlayerState:

	def getPublicState():


	def getOutcome(self):
		#Returns (player1reward,player2reward). NOTE reward is the pot (if game won) or zero the costs of the bets over the course of the game
		#are NOT subtracted
		if self.finished:
			if playerfolded == 1:
				return [0,self.pot]
			elif playerfolded == 2:
				return [self.pot,0]
			elif cards["player1"] == cards["public"]:
				return [self.pot,0]
			elif cards["player2"] == cards["public"]:
				return [0,self.pot]
			elif cards["player1"] < cards["player2"]:
				return [self.pot,0]
			elif cards["player2"] > cards["player1"]:
				return [self.pot,0]
			else:
				return [self.pot/2,self.pot/2]
		else:
			return None

	def getPlayerStates(self):
		pass

	def getPublicHistory(self):
		pass
				
	def isFinished(self):
		return self.finished

	def action(self,strategy):
    	"""Randomly select action from strategy.
    	Args:
        	strategy (list(float)): Strategy of the node
    	Returns:
        	Chosen action, bet size """
    	choice = random.random()
    	action = 2
    	betAmount = 0
    	probability_sum = 0
    	for i in range(3):
        	action_probability = strategy[i]
        	if action_probability == 0:
            	continue
        	probability_sum += action_probability
        	if choice < probability_sum:
            	action = i
        if action ==2:
        	self.finished = True
        	self.playerfolded = self.player
        elif raisesInRound == 2:
        	betAmount = self.bet
        	self.endRound()
        else:
        	if action == 1:
        		if raisesInRound==1:
        			betAmount = self.bet
        			self.endRound()	
        	if action == 0:
        		if raisesInRound==1:
        			betAmount = 2*self.bet
        		else:
        			betAmount = self.bet
        		raisesInRound += 1

        pot +=betAmount
        return action,betAmount



