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
		allPlayersCards = self.game.getPlayerStates()
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
		gameData = { "input" : np.zeros((len(cache), self.game.params["historySize"] + self.game.params["handSize"] + self.game.params["publicCardSize"])),
					"estimTarget" : np.zeros((len(cache),self.game.params["handSize"])),
					"policyTarget": np.zeros((len(cache),self.game.params["actionSize"])),
					"valuesTarget" : np.zeros((len(cache),self.game.params["valueSize"])) }


		for i in range(len(cache)):
			dict = cache[i]
			v = (v[dict["player"]] - dict["moneyBet"])/dict["pot"]
			gameData["input"][i,:] = (self.nnets.preprocessInput(dict["playerCard"],dict["publicHistory"],dict["publicCard"]))
			gameData["estimTarget"][i,:] = (dict["opponentCard"])
			gameData["policyTarget"][i,:] = (dict["treeStrategy"])
			gameData["valuesTarget"][i,:]= v

		return gameData

	def cleanTrees():
		for tree in self.trees:
			tree.cleanTree()

	def setSimulationParams(self, newNumMCTSSims, newEta):
		self.eta=newEta
		self.numMCTSSims=newNumMCTSSims
		for tree in self.trees:
			tree.setNumSimulations(newNumMCTSSims)
