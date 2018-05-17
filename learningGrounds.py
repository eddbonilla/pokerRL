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

def getGameParams():
	"""
	Returns the game parameters
	"""
	dummyGame=LeducGame()
	historySize=dummyGame.getPublicHistory().size
	handSize=dummyGamey.getPlayerCard().size
	publicCardSize= 1 # I want a method for this -E
	actionSize=dummyGame.getActionSize()
	valueSize=1 #I want a method for this-E
	gameParams={
	"historySize":historySize,
	"handSize": handSize,
	"publicCardSize": publicCardSize,
	"actionSize": actionSize,
	"valueSize": valueSize
	}
	return historySize,handSize,publicCardSize,actionSize,valueSize

def trainOne(sess,nnets,gameParams) #train with only 1 instance (for now) It is a draft of what the code should be
	
	nnInputsReservoir=[]
	opponentCardsReservoir=[]
	policiesReservoir=[]
	valuesReservoir=[]
	for j in range(epochs)
		for i in range(IterPerEpoch)
			newInput, newOppCard, newP, newV = selfPlay(eta=0.3, numMCTSSims=100,nnets,cpuct=1)
			nnInputsReservoir.append(newInput)
			opponentCardsReservoir.append(newOppCard)
			policiesReservoir.append(newP)
			valuesReservoir.append(newV)
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