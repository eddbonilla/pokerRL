import math
import numpy as np
import tensorflow as tf
import copy
from model_tunable import nnets
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
from leduc import LeducGame
#from leduc_c import LeducGame
from learningGrounds_tunable import Training
from selfPlay import selfPlay
import time


#Generate log uniform distribution:


def log_uniform(size):
	unif = list( -3*np.random.random_sample(size))
	exp_unif = [10**r for r in unif]

	return exp_unif


hyperparams = {"alpha": 10., "lmbda": log_uniform(10), "numMCTSsims": 200, "batchsize": 128, "batchesPerTrain": 1024 }



#sample hyperparameters and do training




Exploitability_final = []

for i in range(len(hyperparams["lmbda"])):

	print("lambda = " + str(hyperparams["lmbda"][i]))
	A = Training(hyperparams["lmbda"][i].astype('float32'))
	Exploitability = A.doTraining(200)
	Exploitability_final.append(Exploitability)





"""
A = Training(0.005)
A.doTraining(200)
"""

