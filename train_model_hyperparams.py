import math
import numpy as np
import tensorflow as tf
import copy
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from leduc import LeducGame
#from leduc_c import LeducGame
from learningGrounds import Training
from selfPlay import selfPlay
import time


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



hyperparams = {"alpha": (0.001,10), "lmbda": (0.001,1), "tempDecayRate": (0.001,0.01), "gLearn": (0.0001,1), "fLearn": (0.0001,1)} 


hyperparams2 = {"numMCTSsims": 200, "batchSize": 128, "batchesPerTrain": 1024, "gamesPerUpdateNets": 128, "stepsToIncreaseNumSimulations": 20}


hyp_1 = {}


for key in hyperparams:
	if key == "tempDecayRate":
		hyp_1[key] = log_uniform_sample_plus(hyperparams[key],1)
	else:
		hyp_1[key] = log_uniform_sample(hyperparams[key])


hyp2 = dict(hyp_1, **hyperparams2)
print(hyp2)



A = Training(hyp=hyp2)
A.doTraining(200)




