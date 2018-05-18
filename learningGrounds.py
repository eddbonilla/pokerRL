#import threading #We are thinking about asychronous
import tensorflow as tf
import sys
import random
import numpy as np
import time
import threading
from keras import backend as K
from model import nnets
from leduc import LeducGame
from selfPlay import selfPlay
class Training:

	def __init__(self,maxMemory):
		self.N = 0
		self.numShuffled = 0
		self.unShuffledFraction = 0.03
		self.maxMemory = maxMemory
		self.gameParams = LeducGame.params
		self.inputsReservoir = np.zeros((maxMemory, self.gameParams["historySize"] + self.gameParams["handSize"] + self.gameParams["publicCardSize"]))
		self.opponentsCardsReservoir = np.zeros((maxMemory, self.gameParams["handSize"]))
		self.policiesReservoir = np.zeros((maxMemory,self.gameParams["actionSize"]))
		self.valuesReservoir = np.zeros((maxMemory,self.gameParams["valueSize"]))
		self.gamesPerUpdateNets = 128
		compGraph = tf.Graph()
		with compGraph.as_default(), tf.Session() as sess:
			K.set_session(sess)
			self.nnets=nnets(sess,self.gameParams, alpha=1.)
			#saver = tf.train.Saver() #This is probably good practice
			sess.run(tf.global_variables_initializer())
			self.selfPlay = selfPlay(eta=[0.3,0.3],game=LeducGame(), nnets = self.nnets, numMCTSSims=100,cpuct=1)
			#self.playGames()

	def addToReservoirs(self,inputs,opponentCards,policies,vs):
		k = inputs.shape[0]
		assert k == opponentCards.shape[0]
		assert k == policies.shape[0]
		assert k == vs.shape[0]
		if self.N + k < self.maxMemory:
			self.inputsReservoir[self.N:self.N+k, :] = inputs
			self.opponentsCardsReservoir[self.N:self.N+k, :] = opponentCards
			self.policiesReservoir[self.N:self.N+k, :] = policies
			self.valuesReservoir[self.N:self.N+k, :] =vs
		elif self.N >= self.maxMemory:
			keep_prob =  float(self.maxMemory)/(self.N+k)
			keep_masks = np.random.rand((k)) < keep_prob
			replacements = np.random.randint(0,self.maxMemory, size = (np.sum(keep_masks)))
			self.inputsReservoir[replacements,:] = inputs[keep_masks,:]
			self.opponentsCardsReservoir[replacements,:] = opponentCards[keep_masks,:]
			self.policiesReservoir[replacements,:] = policies[keep_masks,:]
			self.valuesReservoir[replacements,:] = vs[keep_masks,:]
		else:
			numLeft = self.maxMemory - self.N 
			self.inputsReservoir[self.N:self.maxMemory, :] = inputs[0:numLeft]
			self.opponentsCardsReservoir[self.N:self.maxMemory, :] = opponentCards[0:numLeft]
			self.policiesReservoir[self.N:self.maxMemory, :] = policies[0:numLeft]
			self.valuesReservoir[self.N:self.maxMemory, :] =vs[0:numLeft]

			numReplace = k - numLeft
			keep_prob =  float(self.maxMemory)/(self.maxMemory+numReplace)
			keep_masks = np.random.rand((k)) < keep_prob
			keep_masks[0:numLeft] = 0
			replacements = np.random.randint(0,self.maxMemory, size = (np.sum(keep_masks)))
			self.inputsReservoir[replacements,:] = inputs[keep_masks,:]
			self.opponentsCardsReservoir[replacements,:] = opponentCards[keep_masks,:]
			self.policiesReservoir[replacements,:] = policies[keep_masks,:]
			self.valuesReservoir[replacements,:] = vs[keep_masks,:]
		self.N += k

	def shuffleReservoirs():
		np.shuffle(self.inputsReservoir)
		np.shuffle(self.opponentsCardsReservoir)
		np.shuffle(self.policiesReservoir)
		np.shuffle(self.valuesReservoir)


	def playGames(self): 
		for i in range(self.gamesPerUpdateNets):
			self.addToReservoirs(self.selfPlay.runGame())
		if (self.N - self.numShuffled)/self.N > self.unShuffledFraction:
			self.shuffleReservoirs()
