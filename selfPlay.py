import math
import numpy as np
from MCTS import MCTS

class selfPlay:
	def __init__(self,game, eta, nnets,numMCTSSims=100,cpuct =1):
		self.game=game
		self.trees = [MCTS(nnets, numMCTSSims, cpuct), MCTS(nnets, numMCTSSims, cpuct)]             #Index labels player
		self.eta=eta # array with the probabilities of following the average strategy for each player
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct
		self.nnets=nnets

	def runGame(self):
		self.game.resetGame()
		cache = []
		ante = self.game.getAnte()
		v = np.zeros(2)
		v-=ante
		#self.cleanTrees()             clean trees each game if we want
		while not self.game.isFinished():

			player = self.game.getPlayer()
			print(player)

			averageStrategy, treeStrategy = self.trees[player].strategy(self.game)
			print("avStrat =" + str(averageStrategy) + "\n treeStrat =" + str(treeStrategy))
			strategy = (1-self.eta[player])*averageStrategy + self.eta[player] * treeStrategy 
			dict = {
					"treeStrategy" :treeStrategy,
					"player": player,
					"publicHistory": self.game.getPublicHistory(),
					"publicCard"  : self.game.getPublicCard(),
					"playerCard"  : self.game.getPlayerCard(),
					"opponentCard"    : self.game.getOpponentCard(),
					"pot"         : self.game.getPot(),
					"moneyBet"    : v[player]
					}
			cache.append(dict)
			action,bet = self.game.action(strategy)
			v[player]-= bet
			print(action,bet,v)
		v += self.game.getOutcome()
		print(v)
		inputs = np.zeros((len(cache), self.game.params["historySize"] + self.game.params["handSize"] + self.game.params["publicCardSize"]))
		opponentCards = np.zeros((len(cache),self.game.params["handSize"]))
		policies = np.zeros((len(cache),self.game.params["actionSize"]))
		vs = np.zeros((len(cache),self.game.params["valueSize"]))


		for i in range(len(cache)):
			dict = cache[i]
			v = (v[dict["player"]] - dict["moneyBet"])/dict["pot"]
			inputs[i,:] = (self.nnets.preprocessInput(dict["playerCard"],dict["publicHistory"],dict["publicCard"]))
			opponentCards[i,:] = (dict["opponentCard"])
			policies[i,:] = (dict["treeStrategy"])
			vs[i,:]= v

		return inputs, opponentCards, policies, vs

	def cleanTrees():
		for tree in self.trees:
			tree.cleanTree()

	def setSimulationParams(self, newNumMCTSSims, newEta):
		self.eta=newEta
		self.numMCTSSims=newNumMCTSSims
		for tree in self.trees:
			tree.setNumSimulations(newNumMCTSSims)

	def testGame(self,numTests):
		testPlayer=0; #Id of the player we are testing
		v_TR=0.
		v_TA=0.
		v_AR=0.
		randomStrategy=np.ones(self.game.params["actionSize"],dtype=float)/self.game.params["actionSize"]
		for j in range(3):
			for i in range(numTests):
				self.game.resetGame()
				ante = self.game.getAnte()
				v = np.array([-ante,-ante],dtype = float)
				while not self.game.isFinished():

					player = self.game.getPlayer()
					averageStrategy, treeStrategy = self.trees[player].strategy(self.game)
					if player == testPlayer:
						if j == 2:
							strategy =averageStrategy
						else:
							strategy = treeStrategy
					else:
						if j == 1:
							strategy= averageStrategy
						else:
							strategy=randomStrategy
					#print(str(player)+" j= "+str(j)+" "+str(strategy))
					action,bet = self.game.action(strategy)
					v[player]-= bet
				v += self.game.getOutcome()
				if j == 0: v_TR+=v[testPlayer]/numTests
				if j == 1: v_TA+=v[testPlayer]/numTests
				if j == 2: v_AR+=v[testPlayer]/numTests
		return v_TR, v_TA, v_AR

