#import threading #We are thinking about asychronous
import tensorflow as tf
import sys
import random
import numpy as np
import time
import threading
from keras import backend as K
from model import nnets
from leduc_c import LeducGame
#from leduc import LeducGame
from selfPlay import selfPlay
import time

class Training:

	def __init__(self,maxPolicyMemory = 1000000, maxValueMemory = 100000,hyp=None):
		self.pN = 0
		self.vN = 0
		self.numShuffled = 0
		self.unShuffledFraction = 0.005 #Maximum fraction of unshuffled data llowed to be in the reservoir
		self.maxPolicyMemory = maxPolicyMemory
		self.maxValueMemory = maxValueMemory

		self.gameParams = {"inputSize" : 30, "historySize" : 24, "handSize" : 3, "actionSize" : 3, "valueSize": 1}

		self.pReservoirs = {"input" : np.zeros((maxPolicyMemory, self.gameParams["inputSize"])),
							"policyTarget" : np.zeros((maxPolicyMemory,self.gameParams["actionSize"]))
							}

		self.vReservoirs = { "playerCard" : np.zeros((maxValueMemory, self.gameParams["handSize"])),
							 "publicData" : np.zeros((maxValueMemory, self.gameParams["historySize"] + self.gameParams["handSize"])),
							 "valuesTarget" : np.zeros((maxValueMemory,self.gameParams["valueSize"])),
							 "estimTarget" : np.zeros((maxValueMemory, self.gameParams["handSize"]))}
		
		self.randState = np.random.RandomState()
		compGraph = tf.Graph()
		compGraph.as_default()
		self.sess= tf.Session()
		K.set_session(self.sess)

		if hyp !=None: #Get parameters form dictionary
			self.gamesPerUpdateNets = hyp["gamesPerUpdateNets"]
			self.batchSize = hyp["batchSize"]
			self.batchesPerTrain = hyp["batchesPerTrain"]
			self.stepsToIncreaseNumSimulations=hyp["stepsToIncreaseNumSimulations"]
			self.nnets=nnets(self.sess,gameParams=self.gameParams,hyp=hyp)
		else:
			self.gamesPerUpdateNets = 128 
			self.batchSize = 128
			self.batchesPerTrain = 1024
			self.stepsToIncreaseNumSimulations=20
			self.nnets=nnets(self.sess,gameParams=self.gameParams)
		
		self.saver = tf.train.Saver() #This is probably good practice
		self.sess.run(tf.global_variables_initializer())
		self.selfPlay = selfPlay(eta=[0.1,0.1],game=LeducGame(), nnets = self.nnets)
		

	def doTraining(self,steps):
		for i in range(steps):
			start = time.time()
			self.playGames()
			postGames = time.time()
			if i%self.stepsToIncreaseNumSimulations==0:
				for tree in self.selfPlay.trees:
					tree.increaseNumSimulations()
			if i%10==0:
				history = np.zeros((2,2,3,2))
				print("Exploitability =" + str(self.selfPlay.trees[0].findAnalyticalExploitability()))
				print("Jack p,v: "+ str(self.nnets.policyValue([1,0,0], history, np.zeros(3))))
				print("Queen p,v: "+ str(self.nnets.policyValue([0,1,0], history, np.zeros(3))))
				print("King p,v: "+ str(self.nnets.policyValue([0,0,1], history, np.zeros(3))))
				history[1,0,0,0] = 1
				print("If op raised, op cards:" + str(self.nnets.estimateOpponent(history,np.zeros(3))))
				print("vN = "+str(self.vN) + ", pN = " +str(self.pN))
			self.selfPlay.cleanTrees()
			prenets = time.time()
			for j in range(self.batchesPerTrain):
				expiredIterator = self.nnets.trainOnMinibatch()
				if expiredIterator is not None:
					self.initIterator(expiredIterator)

			end = time.time()
			if i%10==0:
				print(str(i) + ", selfPlay time = "+str(postGames - start) + ", nnet training time = "+str(end - prenets))
				
		return self.selfPlay.trees[0].findAnalyticalExploitability() #Want to minimize final exploitability after training when sampling over hyperparameters -D
				#print("cost = " + str(self.nnets.compute_cost_alpha()))
		#self.sess.close()

	def addToReservoirs(self,newData):
		pData, vData = newData
		pk = len(pData["input"])
		if pk > 0:
			for key in self.pReservoirs:
				pData[key] = np.array(pData[key])
				assert pk == pData[key].shape[0]
		
			if self.pN + pk < self.maxPolicyMemory:
				for key in self.pReservoirs:
					self.pReservoirs[key][self.pN:self.pN+pk, :] = pData[key]
			elif self.pN >= self.maxPolicyMemory:
				keep_prob =  float(self.maxPolicyMemory)/(self.pN+pk)
				keep_masks = np.random.rand((pk)) < keep_prob
				replacements = np.random.randint(0,self.maxPolicyMemory, size = (np.sum(keep_masks)))
				
				for key in self.pReservoirs:
					self.pReservoirs[key][replacements,:] = pData[key][keep_masks,:]

			else:
				numLeft = self.maxPolicyMemory - self.pN 

				for key in self.pReservoirs:
					self.pReservoirs[key][self.pN:self.maxPolicyMemory, :] = pData[key][0:numLeft]

				numReplace = pk - numLeft
				keep_prob =  float(self.maxMemory)/(self.maxMemory+numReplace)
				keep_masks = np.random.rand((pk)) < keep_prob
				keep_masks[0:numLeft] = 0
				replacements = np.random.randint(0,self.maxPolicyMemory, size = (np.sum(keep_masks)))
				for key in self.pReservoirs:
					self.pReservoirs[key][replacements,:] = pData[key][keep_masks,:]
			self.pN += pk

		vk = len(vData["valuesTarget"])
		#Overwritten data input so that we can feed only recent data to nnets without having to copy arrays
		if self.vN + vk < 1.5*self.maxValueMemory and self.vN >= self.maxValueMemory:
			vN = int(1.5*self.maxValueMemory) - self.vN - vk

		#Normal data input
		else:
			vN = self.vN % self.maxValueMemory

		for key in self.vReservoirs:
			vData[key] = np.array(vData[key])
			assert vk == vData[key].shape[0]
		vData["valuesTarget"] = np.reshape(vData["valuesTarget"],(vk,1))
		if vN + vk <= self.maxValueMemory:
			for key in self.vReservoirs:
				self.vReservoirs[key][vN:vN+vk, :] = vData[key]
		else:
			numLeft = self.maxValueMemory - vN 

			for key in self.vReservoirs:
				self.vReservoirs[key][vN:self.maxValueMemory, :] = vData[key][0:numLeft]
				self.vReservoirs[key][0:(vk - numLeft), :] = vData[key][numLeft:vk]


		self.vN += vk
		

	def shufflePReservoirs(self):
		start = time.time()
		seed = random.randint(0,1000000)
		for key in self.pReservoirs:
			self.randState.seed(seed)
			self.randState.shuffle(self.pReservoirs[key][0:self.pN,:])
		self.numShuffled = self.pN
		end = time.time()
		#print("shuffle time = " + str(end - start))

	def saveSession(self,checkpointPath):
		self.saver.save(self.sess,checkpointPath)

	def initIterator(self, name):
		if name == "pIterator":
			if self.pN < self.maxPolicyMemory:
				shortenedPReservoirs = {}
				for key in self.pReservoirs:
					shortenedPReservoirs[key] =self.pReservoirs[key][0:self.pN,:]
			else:
				shortenedPReservoirs = self.pReservoirs
			self.nnets.initialisePIterator(shortenedPReservoirs)

		if name == "vIterator":
			if self.vN <= self.maxValueMemory:
				shortenedVReservoirs = {}
				for key in self.vReservoirs:
					shortenedVReservoirs[key] =self.vReservoirs[key] [int(self.vN/2):self.vN,:]
			elif self.vN < 1.5*self.maxValueMemory:
				shortenedVReservoirs = {}
				for key in self.vReservoirs:
					shortenedVReservoirs[key] =self.vReservoirs[key] [(int(1.5*self.maxValueMemory) - self.vN):self.maxValueMemory,:]
			else:
				shortenedVReservoirs = self.vReservoirs
			self.nnets.initialiseVIterator(shortenedVReservoirs)
			#print(self.vN)

	def playGames(self): 
		
		oldvN = self.vN
		oldpN = self.pN
		
		for i in range(self.gamesPerUpdateNets):
			self.addToReservoirs(self.selfPlay.runGame())
		
		if (self.pN - self.numShuffled)/self.pN > self.unShuffledFraction:
			self.shufflePReservoirs()

		for tree in self.selfPlay.trees:
			tree.reduceTemp()


	def closeSession(self):
		self.sess.close()
