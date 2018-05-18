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
		self.reservoirs = {"input" : np.zeros((maxMemory, self.gameParams["historySize"] + self.gameParams["handSize"] + self.gameParams["publicCardSize"])),
							"estimTarget" : np.zeros((maxMemory, self.gameParams["handSize"])),
							"policyTarget" : np.zeros((maxMemory,self.gameParams["actionSize"])),
							"valuesTarget" : np.zeros((maxMemory,self.gameParams["valueSize"])) }
		self.gamesPerUpdateNets = 128
		compGraph = tf.Graph()
		with compGraph.as_default(), tf.Session() as sess:
			K.set_session(sess)
			self.nnets=nnets(sess,self.gameParams, alpha=1.)
			#saver = tf.train.Saver() #This is probably good practice
			sess.run(tf.global_variables_initializer())
			self.selfPlay = selfPlay(eta=[0.3,0.3],game=LeducGame(), nnets = self.nnets, numMCTSSims=100,cpuct=1)
			self.playGames()

	def addToReservoirs(self,newData):
		k = newData["input"].shape[0]
		for key in reservoirs:
			assert k == newData[key].shape[0]
		if self.N + k < self.maxMemory:
			for key in self.reservoirs:
			self.reservoirs[key][self.N:self.N+k, :] = newData
		elif self.N >= self.maxMemory:
			keep_prob =  float(self.maxMemory)/(self.N+k)
			keep_masks = np.random.rand((k)) < keep_prob
			replacements = np.random.randint(0,self.maxMemory, size = (np.sum(keep_masks)))
			
			for key in self.reservoirs:
				self.reservoirs[key][replacements,:] = newData[key][keep_masks,:]

		else:
			numLeft = self.maxMemory - self.N 

			for key in self.reservoirs:
				self.reservoirs[key][self.N:self.maxMemory, :] = newData[key][0:numLeft]

			numReplace = k - numLeft
			keep_prob =  float(self.maxMemory)/(self.maxMemory+numReplace)
			keep_masks = np.random.rand((k)) < keep_prob
			keep_masks[0:numLeft] = 0
			replacements = np.random.randint(0,self.maxMemory, size = (np.sum(keep_masks)))
			for key in self.reservoirs:
				self.reservoirs[key][replacements,:] = newData[key][keep_masks,:]
		self.N += k

	def shuffleReservoirs():
		for key in self.reservoirs:
			np.shuffle(self.reservoirs[key])



	def playGames(self): 
		for i in range(self.gamesPerUpdateNets):
			self.addToReservoirs(self.selfPlay.runGame())
		if (self.N - self.numShuffled)/self.N > self.unShuffledFraction:
			self.shuffleReservoirs()
