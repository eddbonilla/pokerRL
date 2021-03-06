import math
import numpy as np
import tensorflow as tf
from learningGrounds import Training
import time
import collections



#Generate log uniform distribution:


def log_uniform_sample(interval):

	a = np.log10(min(interval))
	b = np.log10(max(interval))

	unif = (b-a)*np.random.random_sample()+a
	exp_unif = 10**unif

	return exp_unif

def log_uniform_sample_plus(interval,plus):

	a = np.log10(min(interval))
	b = np.log10(max(interval))

	unif = (b-a)*np.random.random_sample()+a
	exp_unif_plus = 10**unif + plus

	return exp_unif_plus


#file = open("./hyperparameter_check/test4.txt", "a")

hyperparams = collections.OrderedDict([("alpha", (0.001,10)), ("lmbda", (0.001,1)), ("tempDecayRate", (0.001,0.01)), ("gLearn", (0.0001,1)), ("fLearn", (0.0001,1))]) #ordered dictionary

hyperparams2 = collections.OrderedDict([("numMCTSsims", 200), ("batchSize", 128), ("batchesPerTrain", 1024), ("gamesPerUpdateNets", 128), ("stepsToIncreaseNumSimulations", 20)]) #ordered dictionary

hyperparams3 = {"alpha": (0.001,10), "lmbda": (0.001,1),"tempDecayRate": (0.0001,0.01),"gLearn": (0.0001,1), "fLearn": (0.0001,1)} #unordered dictionary
hyperparams4 = {"numMCTSsims": 200, "batchSize": 128, "batchesPerTrain": 1024, "gamesPerUpdateNets": 128, "stepsToIncreaseNumSimulations": 20} #unordered dictionary


hypList = []

def generateDictionaryList(numIterations):
	for i in range(numIterations):
		hyp_1 = {}
		for key in hyperparams3:
			if key == "tempDecayRate":
				hyp_1[key] = log_uniform_sample_plus(hyperparams[key],1)
			else:
				hyp_1[key] = log_uniform_sample_plus(hyperparams[key],0)


		hyp_1.update(hyperparams4)
		
		hypList.append(hyp_1)

	return hypList

#print(generateDictionaryList(2))

def hyperparameterSearch(numIterations,numTrainSteps):

	hypList = generateDictionaryList(numIterations)


	for i in range(numIterations):
		print(hypList[i])
		A = Training(directory='logs'+str(i),hyp=hypList[i])
		finalExploitability,minExploitability= A.doTraining(numTrainSteps)
		A.closeSession()
		print(i)
		#file.write(str(finalExploitability)+" "+str(minExploitability)+" "+str(numTrainSteps)+ " "+str(hyp_1)+"\n")
		


#B = Training(hyp = collections.OrderedDict([('alpha', 4.784406328533163), ('lmbda', 0.0027503368223086894), ('tempDecayRate', 1.0062407845204762), ('gLearn', 0.12800828394213012), ('fLearn', 0.00017930828290086467), ('numMCTSsims', 200), ('batchSize', 128), ('batchesPerTrain', 1024), ('gamesPerUpdateNets', 128), ('stepsToIncreaseNumSimulations', 20)]))
#B.doTraining(3)


hyperparameterSearch(2,10)









