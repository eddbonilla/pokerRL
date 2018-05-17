import math
import numpy as np
import copy
from leduc import LeducGame
from MCTS import MCTS
class nnet:
	def policyValue(self,a,b,c):
		p=[1.,1.,1.]
		return p/np.sum(p) , 0.5

	def estimateOpponent(self,a,b,c):
		g=[1.,1.,1.]
		return g/np.sum(g)

game=LeducGame()
net=nnet()
tree=MCTS(net, numMCTSSims=2000, cpuct=1)
boring, strat=tree.strategy(game)

print(strat)
