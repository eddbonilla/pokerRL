"""
These are utility functions to be called when the neural networks are trained. It contains all the forward propagations, backprop will be done when the models are trained. I put in some placeholder values for X and Y so as to be able to test forward propagation.
"""

import numpy as np
import tensorflow as tf
import math




X = tf.placeholder(tf.float32,shape=(None,784))
Y = tf.placeholder(tf.float32,shape=(None,10))


####CREATE PLACEHOLDERS######



######INITIALIZE PARAMETERS##########

def initialize_parameters_deep(layer_dims):
	"""
	Initialize the parameters of our L-layer neural network"
	"""

	parameters = {}
	L = len(layer_dims)

	for l in range(1,L):
		parameters['W'+str(l)] = tf.get_variable("W"+str(l),shape=[layer_dims[l-1],layer_dims[l]],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
		parameters['b'+str(l)] = tf.get_variable("b"+str(l),shape=[1,layer_dims[l]],initializer=tf.zeros_initializer())

		assert(parameters['W' + str(l)].shape == (layer_dims[l-1], layer_dims[l]))
		assert(parameters['b' + str(l)].shape == (1,layer_dims[l]))

	return parameters

parameters = initialize_parameters_deep([784,4,3,17,24])

"""
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("W3 = " + str(parameters["W3"]))
print("b3 = " + str(parameters["b3"]))
print("W4 = " + str(parameters["W4"]))
print("b4 = " + str(parameters["b4"]))
"""


########FORWARD PROPAGATION############
def linear_relu_forward(A_prev,W,b):
	"""
	single step of relu
	"""
	Z = tf.matmul(A_prev,W)+b
	A = tf.nn.relu(Z)

	return A

def linear_softmax_forward(A_prev,W,b):
	"""
	single step of softmax
	"""

	Z = tf.matmul(A_prev,W)+b
	A = tf.nn.softmax(Z)

	return Z,A


###Forward propagate from public history
def L_model_forward(public_history,parameters):

	A = public_history
	L = len(parameters) // 2

	for l in range(1,L):
		A_prev = A
		A = linear_relu_forward(A,parameters['W'+str(l)],parameters['b'+str(l)])
	

	ZL,opp_priv_info = linear_softmax_forward(A,parameters['W' + str(L)],parameters['b' + str(L)]) 
	

	return ZL,opp_priv_info

#_,AL = L_model_forward(X,parameters)
ZL,_ = L_model_forward(X,parameters)
#print(ZL)

###############CREATE MINIBATCHES########################

###Create random data set to test
X_assess = tf.random_normal(shape=(148,12288))
Y_assess = tf.random_normal(shape=(148,1))

"""
sess1 = tf.Session()
m = sess1.run(tf.cast(X_assess.shape[0],tf.int32))
permutation = list(np.random.permutation(m))

print(sess1.run(X_assess))
print(sess1.run(X_assess))
"""




#m = sess1.run(tf.cast(X_assess.shape[0],tf.int32))
#print(list(np.random.permutation(m)))



def random_mini_batches(X,Y,mini_batch_size):


	#Creates a list of random minibatches of X,Y. Tensorflow seems to have an inbuilt minibatch sampler, but I'm just building one based on the programming assigments. We can adjust this later



	sess1 = tf.Session()
	m = sess1.run(tf.cast(X_assess.shape[0],tf.int32))	

	mini_batches = []
	#permutation = list(np.random.permutation(m))
	#print(permutation)
	shuffled_X = sess1.run(tf.random_shuffle(X))
	shuffled_Y = sess1.run(tf.random_shuffle(Y))

	
	num_complete_minibatches = math.floor(m/mini_batch_size)

	for k in range(0, num_complete_minibatches):
        
		mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
		mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
      
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_batch_size != 0:
        
		mini_batch_X = shuffled_X[0:(m - mini_batch_size*num_complete_minibatches),:]
		mini_batch_Y = shuffled_Y[0:(m - mini_batch_size*num_complete_minibatches),:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches


mini_batches = random_mini_batches(X_assess,Y_assess,mini_batch_size = 64)
print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))




















