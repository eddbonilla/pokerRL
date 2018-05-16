import math
import numpy as np
import copy
from leduc import LeducGame
from MCTS import MCTS
class fnet():
	def predict(self,a,b,c,d):
		return [0.5, 0.5, 0] , 1.5

class gnet():
	def predict(self,a,b,c):
		g=[0,1,0]
		return g 

game=LeducGame()
f=fnet()
g=gnet()
tree=MCTS(game, f, g, numMCTSSims=10, cpuct=1)
strat=tree.treeStrategy(1,2,3)

print(strat)
