import math
import numpy as np
import tensorflow as tf
import copy
from model import nnets
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from leduc import LeducGame

from selfPlay import selfPlay

"""
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
"""


with tf.Session() as session:
	K.set_session(session)
	params = {"historySize" : 24, "handSize" : 3, "publicCardSize" : 3, "actionSize" : 3, "valueSize": 1}
	A=tf.placeholder(dtype=tf.float32,shape=(1,2))
	F=nnets(session,params)
	session.run(tf.global_variables_initializer())
	Q=LeducGame()
	B=selfPlay(Q,[0.3,0.5],100,F,1)
	output=B.runGame()
	print("Success?")
