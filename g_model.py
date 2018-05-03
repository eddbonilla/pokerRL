"""
This will eventually become the neural net g. For now we are building a deep neural net for the mnist data set, with a softmax output. 
"""

import tensorflow as tf

# Load dataset using tensorflows mnist API
### START CODE HERE (Question 1) ###
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
### END CODE HERE (Question 1) ###
#print(mnist.train.images.shape)
#print(mnist.test.images.shape)
#print(mnist.test.labels.shape)

# Define placeholder input data  matrix and for the labels
### START CODE HERE (Question 2) ###


X = tf.placeholder(tf.float32,shape=(None,784))
Y = tf.placeholder(tf.float32,shape=(None,10))



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

parameters = initialize_parameters_deep([784,4,3,17])

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("W3 = " + str(parameters["W3"]))
print("b3 = " + str(parameters["b3"]))


########FORWARD PROPAGATION############
def linear_relu_forward(A_prev,W,b):
	""""
	single step of relu
	"""
	Z = tf.matmul(A_prev,W)+b
	A = tf.nn.relu(Z)

	return A

def linear_softmax_forward(A_prev,W,b):

	Z = tf.matmul(A_prev,W)+b
	A = tf.nn.softmax(Z)

	return A

def L_model_forward(X,parameters):

	A = X
	L = len(parameters) // 2

	for l in range(1,L):
		A_prev = A
		A = linear_relu_forward(A,parameters['W'+str(l)],parameters['b'+str(l)])

	AL = linear_softmax_forward(A,parameters['W' + str(L)],parameters['b' + str(L)])

	return AL

AL = L_model_forward(X,parameters)

print(AL)



"""

W1 = tf.get_variable("W1",shape = [784,100],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1",shape=[100],initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2",shape = [100,10],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b2 = tf.get_variable("b2",shape=[10],initializer=tf.zeros_initializer())

### END CODE HERE (Question 3) ###

# Define your model's tensorflow graph
### START CODE HERE (Question 4) ###

Z1 = tf.matmul(X,W1) + b1
A1 = tf.nn.relu(Z1)
Z2 = tf.matmul(A1,W2) + b2
A2 = tf.nn.softmax(Z2)


#Split A2

p,v = tf.split(A2,[9,1],1)
print(p.shape)


#Split Z2

z1split, z2split = tf.split(Z2,[9,1],1)
print(z1split.shape)
print(z2split.shape)


### END CODE HERE (Question 4) ###

# Compute the cost function
### START CODE HERE (Question 5) ###

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z2,labels=Y))

### END CODE HERE (Question 5) ###

# Define accuracy metric
### START CODE HERE (Question 6) ###

accuracy2 = tf.metrics.accuracy(tf.argmax(Y, axis=-1),tf.argmax(A2,axis=-1)) #something wrong with this line. ask in OH
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(A2,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### END CODE HERE (Question 6) ###


# Define optimization method, learning rate and the the training step
### START CODE HERE (Question 7) ###

optimizer = tf.train.GradientDescentOptimizer(0.25)
train_step = optimizer.minimize(cost)

### END CODE HERE (Question 7) ###

# Initialize the variables of the graph, create tensorflow session and run the initialization of global variables.
### START CODE HERE (Question 8) ###

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


### END CODE HERE (Question 8) ###

# Implement the Optimization Loop for 100 iterations
### START CODE HERE (Question 9) ###

for i in range(100):
	#Load batches
	batch_X,batch_Y = mnist.train.next_batch(100)
	train_data = {X: batch_X, Y: batch_Y}
	_, cost1 = sess.run([train_step, cost],feed_dict=train_data)
print("Iteration: " + str(i), "training cost " + str(cost1))

"""



