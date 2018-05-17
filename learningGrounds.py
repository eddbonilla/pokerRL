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
from selfPlay import SelfPlay
class Training:

	def __init__(self,maxMemory):
		self.N = 0
		self.maxMemory = maxMemory
		self.gameParams = LeducGame.params
		self.inputsReservoir = np.zeros((maxMemory, self.gameParams["historySize"] + self.gameParams["handSize"] + self.gameParams["publicCardSize"]))
		self.opponentsCardsReservoir = np.zeros((maxMemory, self.gameParams["handSize"]))
		self.policiesReservoir = np.zeros((maxMemory,self.gameParams["actionSize"]))
		self.valuesReservoir = np.zeros((maxMemory,self.gameParams["valueSize"]))

	def addToReservoirs():
		if s

	def trainOne(sess,nnets,gameParams) #train with only 1 instance (for now) It is a draft of what the code should be
	
		for j in range(epochs)
			for i in range(IterPerEpoch)
				self.addToReservoirs(selfPlay(eta=0.3, numMCTSSims=100,nnets,cpuct=1))
				if i%updateTime==0
					minibatch= cutMinibatches(Reservoirs)
					nnets.trainOnMinibatch()
			print(Cost)
			Save(Something)

	def main(_)
		compGraph = tf.Graph()
		with compGraph.as_default(), tf.Session() as sess:
			K.set_session(sess)
			gameParams=getGameParams;
			networks=nnets(sess,gameParams, alpha=1.)
			#saver = tf.train.Saver() #This is probably good practice
			session.run(tf.global_variables_initializer())
			trainOne(sess,nnets,gameParams)