import math
import numpy as np
from MCTS import MCTS

class selfPlay:
	def __init__(self,game, eta, nnets,numMCTSSims=50,cpuct =2,simParams=None):
		self.game=game
		self.trees = [MCTS(nnets, numMCTSSims, cpuct), MCTS(nnets, numMCTSSims, cpuct)]             #Index labels player
		
		if simParams!= None: 
			for tree in self.trees:
				tree.setTreeSearchParams(simParams["TreeSearchParams"])

		self.eta=eta # array with the probabilities of following the average strategy for each player
		self.numMCTSSims=numMCTSSims
		self.cpuct=cpuct
		self.nnets=nnets

	def runGame(self):
		self.game.resetGame()
		cache = { "input" : [],
					"estimTarget" : [],
					"policyTarget": [],
					"valuesTarget" : [],
					"player" : [],
					"pot" : []}
		ante = self.game.getAnte()
		value = np.zeros(2) #harcoded 2 players -E
		value-=ante
		moveCount = 0
		#self.cleanTrees()             clean trees each game if we want
		while not self.game.isFinished():
			if moveCount > 25: #Harcoded max number of moves -E
				print("Stuck in game loop")
				print(cache)
				break
			moveCount += 1

			player = self.game.getPlayer()
			#print(player)

			averageStrategy, treeStrategy = self.trees[player].strategy(self.game)
			#print("avStrat =" + str(averageStrategy) + "\n treeStrat =" + str(treeStrategy))
			strategy = (1-self.eta[player])*averageStrategy + self.eta[player] * treeStrategy 
			strategy /= np.sum(strategy)

			cache["input"].append(self.nnets.preprocessInput(self.game.getPlayerCard(),self.game.getPublicHistory(),self.game.getPublicCard()))
			cache["policyTarget"].append(treeStrategy)
			cache["estimTarget"].append(self.game.getOpponentCard())
			cache["valuesTarget"].append(value[player])
			cache["player"].append(player)
			cache["pot"].append(self.game.getPot())
			#assert np.sum(value) + dict["pot"] == 0
			#print(strategy)
			action,bet = self.game.action(strategy = strategy)
			value[player]-= bet
			#print(action,bet,value)

		value += self.game.getOutcome()
		#assert np.sum(value) == 0
		#print(value)


		for i in range(len(cache["valuesTarget"])):
			cache["valuesTarget"][i] = (value[cache["player"][i]] - cache["valuesTarget"][i])/cache["pot"][i]

		return cache

	def cleanTrees(self):
		for tree in self.trees:
			tree.cleanTree()

	def setSimulationParams(self, newNumMCTSSims, newEta):
		self.eta=newEta
		self.numMCTSSims=newNumMCTSSims
		for tree in self.trees:
			tree.setNumSimulations(newNumMCTSSims)

	def testGame(self,numTests):
		testPlayer=0; #Id of the player we are testing
		v_TC=0.
		v_TA=0.
		v_AC=0.
		checkStrategy=np.array([0,1,0])

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

							strategy=checkStrategy

					#print(str(player)+" j= "+str(j)+" "+str(strategy))
					action,bet = self.game.action(strategy =strategy)
					v[player]-= bet
				v += self.game.getOutcome()
				if j == 0: v_TC+=v[testPlayer]/numTests
				if j == 1: v_TA+=v[testPlayer]/numTests
				if j == 2: v_AC+=v[testPlayer]/numTests

		print("v_TC="+str(v_TC)+"\t"+"v_TA="+str(v_TA)+"\t"+"v_AC="+str(v_AC))
		return v_TC, v_TA, v_AC


