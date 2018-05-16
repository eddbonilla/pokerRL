import math
import numpy as np
import copy
from leduc import LeducGame
from MCTS import MCTS
class nnet():
	def policyValue(self,a,b,c):
		return [1, 0, 0] , 1

	def estimateOpponent(self,a,b,c):
		g=[0,1,0]
		return g 

game=LeducGame()
net=nnet()
tree=MCTS(game, net, numMCTSSims=100, cpuct=1)
strat=tree.treeStrategy(1,2,3)

print(strat)
