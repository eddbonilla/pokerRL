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
import time

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
		self.batchSize = 128
		self.randState = np.random.RandomState()
		self.batchesPerTrain = 3

		compGraph = tf.Graph()
		compGraph.as_default()
		self.sess= tf.Session()
		K.set_session(self.sess)
		self.nnets=nnets(self.sess,self.gameParams, alpha=1.)
		self.saver = tf.train.Saver() #This is probably good practice
		self.sess.run(tf.global_variables_initializer())
		self.selfPlay = selfPlay(eta=[0.3,0.3],game=LeducGame(), nnets = self.nnets, numMCTSSims=50,cpuct=1)

	def doTraining(self,steps):
		for i in range(steps):
			start = time.time()
			self.playGames()
			postGames = time.time()
			for j in range(self.batchesPerTrain):
				self.nnets.trainOnMinibatch()
			end = time.time()
			print(str(i) + ", selfPlay time = "+str(postGames - start) + ", nnet training time = "+str(end - postGames))
		#self.sess.close()

	def addToReservoirs(self,newData):
		k = len(newData["input"])
		for key in self.reservoirs:
			newData[key] = np.array(newData[key])
			assert k == newData[key].shape[0]
		newData["valuesTarget"] = np.reshape(newData["valuesTarget"],(k,1))
		if self.N + k < self.maxMemory:
			for key in self.reservoirs:
				self.reservoirs[key][self.N:self.N+k, :] = newData[key]
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

	def shuffleReservoirs(self):
		shortenedReservoirs = {}
		seed = random.randint(0,1000000)
		for key in self.reservoirs:
			self.randState.seed(seed)
			shortenedReservoirs[key] = self.reservoirs[key][0:self.N,:]
			self.randState.shuffle(shortenedReservoirs[key])

		if self.numShuffled < self.maxMemory:
			self.nnets.initialiseIterator(shortenedReservoirs, self.batchSize)
		self.numShuffled = self.N

	def saveSession(self,checkpointPath):
		self.saver.save(self.sess,checkpointPath)




	def playGames(self): 
		for i in range(self.gamesPerUpdateNets):
			self.addToReservoirs(self.selfPlay.runGame())
		if self.N - self.numShuffled > self.maxMemory or (self.numShuffled < self.maxMemory and (self.N - self.numShuffled)/self.N > self.unShuffledFraction):
			self.shuffleReservoirs()
		self.selfPlay.cleanTrees()

	def closeSession(self):
		self.sess.close()
