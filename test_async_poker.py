#import threading #We are thinking about asynchronous
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
from asyncLearningGrounds import trainingThread
import time
#One idea that might be better is play a couple games to warm up the reservoir first and then sample from it
#another idea is that maybe we want to get rid of very old memories, the long term average should wash them away anyways
treeSearchParams={
"initialNumMCTS":100,
"initialTreeTemperature":1,
"tempDecayRate":1.002, #This is the number dividing the initial temperature
"cpuct":1
}
simParams={
"maxValueMemory": 1048576, #maximum memory of a single  actor learner thread
"maxPolicyMemory":1048576,
"unShuffledFraction":0.001,
"batchesPerTrain":512,
"gamesPerUpdateNets":128,
"batchSize":128,
"eta":[0.1,0.1],
"alpha":40.,
"numStepsEpoch":10,
"treeSearchParams":treeSearchParams,
"printExploitFreq":30,
"printSelfPlayFreq":10
}
global T
T=0
numThreads= 4 #You want this number to be roughly equal to the number of threads in your computer
randomSeed=time.time() #set a specific seed for the RNG to replicate things if needed
def desperateMeasure(trainer,numStepsEpoch):
	global T
	print("hello")
	while T<10000:
		for i in range(10):
			trainer.doTraining(numStepsEpoch)
		T+=1
		print(T)

try:
	g = tf.Graph()
	with g.as_default(), tf.Session() as session:


		K.set_session(session)
		random.seed(randomSeed)
		gameParams=LeducGame.params #Explicit reference to LeducGame to get parameters
		nets=nnets(session,gameParams,simParams["alpha"]) #build the computation graph and all the operations
		session.run(tf.global_variables_initializer()) #initialize all variables

		games = [ LeducGame(seed=random.randint(0,2**32-1)) for i in range(numThreads)] #build different games for each thread
		trainers= [trainingThread(games[i],simParams,nets,threadId=i+1, seed=random.randint(0,2**32-1)) for i in range(numThreads)] #Build different training instances
		actorLearnerThreads = [threading.Thread(target=desperateMeasure,args=(trainers[i],simParams["numStepsEpoch"],)) for i in range(numThreads)] #Target the functions inside the class for multithreading

		for t in actorLearnerThreads:
			print("start")
			t.start()
		while T<10000:
			time.sleep(600)
		print(session)

except(KeyboardInterrupt,SystemExit):
	print("interrupted")
	session.close()