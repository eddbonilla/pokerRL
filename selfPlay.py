import math
import numpy as np
from MCTS import MCTS

class selfPlay:
	def __init__(self,game, eta, numMCTSSims,nnets,cpuct):
		self.game=game
		self.trees = [MCTS(nnets, numMCTSSims, cpuct), MCTS(nnets, numMCTSSims, cpuct)]             #Index labels player
		self.eta=eta # array with the probabilities of following the average strategy for each player
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct

	def runGame(self):
		self.game.resetGame()
		cache = []
		allPlayersCards = self.game.getPlayerStates()
		ante = self.game.getAnte()
		v = np.array([-ante,-ante],dtype = float)
		#self.cleanTrees()             clean trees each game if we want
		while not self.game.isFinished():

			player = self.game.getPlayer()
			print(player)

			averageStrategy, treeStrategy = self.trees[player].strategy(self.game)
			print("avStrat =" + str(averageStrategy) + "\n treeStrat =" + str(treeStrategy))
			strategy = (1-eta[player])*averageStrategy + eta[player] * treeStrategy 
			dict = {
					"treeStrategy" :treeStrategy,
					"player": player,
					"publicHistory": game.getPublicHistory(),
					"publicCard"  : game.getPublicCard(),
					"playerCard"  : game.getPlayerCard(),
					"opponentCard"    : game.getOpponentCard(),
					"pot"         : game.getPot(),
					"moneyBet"    : v[player]
					}
			cache.append(dict)
			action,bet = self.game.action(strategy)
			v[player]-= bet
			print(action,bet,v)
		v += game.getOutcome()
	
		inputs = np.zeros((len(cache), self.game.params["historySize"] + self.game.params["handSize"] + self.game.params["publicCardSize"]))
		opponentCards = np.zeros((len(cache),self.game.params["handSize"]))
		policies = np.zeros((len(cache),self.game.params["actionSize"]))
		vs = np.zeros((len(cache),self.game.params["valueSize"]))


		for i in len(cache):
			dict = cache[i]
			v = (v[dict["player"]] - dict["moneyBet"])/float(dict["pot"])
			inputs[i,:] = (nnets.preprocessInput(dict["playerCard"],dict["publicHistory"],dict["publicCard"]))
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
