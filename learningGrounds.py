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

	def __init__(self,maxPolicyMemory = 1000000, maxValueMemory = 100000):
		self.pN = 0
		self.vN = 0
		self.numShuffled = 0
		self.unShuffledFraction = 0.005
		self.maxPolicyMemory = maxPolicyMemory
		self.maxValueMemory = maxValueMemory
		self.gameParams = LeducGame.params

		self.pReservoirs = {"input" : np.zeros((maxPolicyMemory, self.gameParams["inputSize"])),
							"policyTarget" : np.zeros((maxPolicyMemory,self.gameParams["actionSize"]))
							}

		self.vReservoirs = { "input" : np.zeros((maxValueMemory, self.gameParams["inputSize"])),
							 "valuesTarget" : np.zeros((maxValueMemory,self.gameParams["valueSize"])),
							 "estimTarget" : np.zeros((maxValueMemory, self.gameParams["handSize"]))}
		self.gamesPerUpdateNets = 128
		self.batchSize = 128
		self.randState = np.random.RandomState()
		self.batchesPerTrain = 128

		compGraph = tf.Graph()
		compGraph.as_default()
		self.sess= tf.Session()
		K.set_session(self.sess)
		self.nnets=nnets(self.sess,self.gameParams, alpha=1.)
		self.saver = tf.train.Saver() #This is probably good practice
		self.sess.run(tf.global_variables_initializer())
		self.selfPlay = selfPlay(eta=[0.1,0.1],game=LeducGame(), nnets = self.nnets, numMCTSSims=100,cpuct=2)

	def doTraining(self,steps):
		for i in range(steps):
			start = time.time()
			self.playGames()
			postGames = time.time()
			if i%30==0:
				print("Exploitability =" + str(self.selfPlay.trees[0].findExploitability()))
				print("Jack p,v: "+ str(self.nnets.policyValue([1,0,0], np.zeros(2,2,3,2), np.zeros(3))))
				print("Queen p,v: "+ str(self.nnets.policyValue([0,1,0], np.zeros(2,2,3,2), np.zeros(3))))
				print("King p,v:"+ str(self.nnets.policyValue([0,0,1], np.zeros(2,2,3,2), np.zeros(3))))
			self.selfPlay.cleanTrees()
			prenets = time.time()
			for j in range(self.batchesPerTrain):
				self.nnets.trainOnMinibatch()
			end = time.time()
			if i%10==0:
				print(str(i) + ", selfPlay time = "+str(postGames - start) + ", nnet training time = "+str(end - prenets))
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

		vk = len(vData["input"])
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
		print("shuffle time = " + str(end - start))

	def saveSession(self,checkpointPath):
		self.saver.save(self.sess,checkpointPath)




	def playGames(self): 
		
		oldvN = self.vN
		oldpN = self.pN
		
		for i in range(self.gamesPerUpdateNets):
			self.addToReservoirs(self.selfPlay.runGame())
		
		if (self.pN - self.numShuffled)/self.pN > self.unShuffledFraction:
			self.shufflePReservoirs()
			if oldpN < self.maxPolicyMemory:
				shortenedPReservoirs = {}
				for key in self.pReservoirs:
					shortenedPReservoirs[key] =self.pReservoirs[key][0:self.pN,:]
				self.nnets.initialisePIterator(shortenedPReservoirs)
		
		if oldvN < self.maxValueMemory:
			shortenedVReservoirs = {}
			for key in self.vReservoirs:
				shortenedVReservoirs[key] =self.vReservoirs[key] [self.vN:0:-1,:]
			self.nnets.initialiseVIterator(shortenedVReservoirs)
			#print(self.vN)
		

		for tree in self.selfPlay.trees:
			tree.reduceTemp()


	def closeSession(self):
		self.sess.close()
