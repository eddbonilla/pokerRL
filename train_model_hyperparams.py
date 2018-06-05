import math
import numpy as np
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


file = open("./hyperparameter_check/test1.txt", "a")

hyperparams = collections.OrderedDict([("alpha", (0.001,10)), ("lmbda", (0.001,1)), ("tempDecayRate", (0.001,0.01)), ("gLearn", (0.0001,1)), ("fLearn", (0.0001,1))])

hyperparams2 = collections.OrderedDict([("numMCTSsims", 200), ("batchSize", 128), ("batchesPerTrain", 1024), ("gamesPerUpdateNets", 128), ("stepsToIncreaseNumSimulations", 20)])


def hyperparameterSearch(numIterations,numTrainSteps):
	for i in range(numIterations):
		hyp_1 = collections.OrderedDict()
		for key in hyperparams:
			if key == "tempDecayRate":
				hyp_1[key] = log_uniform_sample_plus(hyperparams[key],1)
			else:
				hyp_1[key] = log_uniform_sample(hyperparams[key])


		hyp_1.update(hyperparams2)
		#print(hyp_1)

		A = Training(hyp=hyp_1)
		finalExploitability,minExploitability= A.doTraining(numTrainSteps)
		A.closeSession()
		print(i)
		file.write(str(finalExploitability)+" "+str(minExploitability)+" "+str(numTrainSteps)+ " "+str(hyp_1)+"\n")


hyperparameterSearch(2,2)


