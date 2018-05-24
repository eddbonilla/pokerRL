import tensorflow as tf
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model
import numpy as np
import functools

#decorator, used to simplify the implementation of the tensorflow graph. Taken from 
#Danijar Hafner's article on the subject
def define_scope(function):
	attribute = '_cache_' + function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			with tf.variable_scope(function.__name__):
				setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator

class nnets:
	"""docstring for nnet"""
	def __init__(self,session,gameParams,layers_f,layers_g,alpha =1.):

		self.sess=session
		self.gameParams=gameParams

		#Create placeholders
		self.rawData={ "input" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["inputSize"])),
						"estimTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["handSize"])),
						"policyTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["actionSize"])),
						"valuesTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["valueSize"])) }
		self.batchSize = tf.placeholder(tf.int64, shape=[])
		
		self.dataset = tf.data.Dataset.from_tensor_slices(self.rawData)
		self.dataset = self.dataset.repeat()
		self.dataset = self.dataset.batch(self.batchSize)
		self.dataset = self.dataset.prefetch(1)
		self.iterator = self.dataset.make_initializable_iterator()

		self.nnetsData =self.iterator.get_next()

		#Properties that we are setting to be constants
		self.alpha=float(alpha)
		self.layers_f = int(layers_f)
		self.layers_g = int(layers_g)

		#Model properties
		self.gModel=None
		self.fModel=None

		#Properties that are actually graph nodes
		self.getLogitsOpponent
		self.getEstimateOpponent
		self.costEstimate
		self.trainEstimate
		self.getLogitsValue
		self.getPolicyValue #sets up the graph and computes the Policy/Value
		self.costPolicyValue
		self.trainPolicyValue

#Functions that set properties during initialization
	@define_scope
	def getLogitsOpponent(self):

		input_size = int(self.gameParams["inputSize"])
		target_size= int(self.gameParams["handSize"])
		inputs = Input(shape=(input_size,))
		model = Dense(output_dim=256, activation='relu')(inputs)
		for i in range(self.layers_g): #adding more layers, maybe for real poker
			model_prev = model
			model = Dense(output_dim=256, activation='relu')(model)
			#print(model.get_shape())
		cards = Dense(output_dim=target_size, activation='linear')(model)
		#print(cards.get_shape())
		self.gModel = Model(input=inputs, output=cards)
		return self.gModel(self.nnetsData["input"])


	@define_scope
	def getEstimateOpponent(self):
		return tf.nn.softmax(self.getLogitsOpponent)


	@define_scope
	def costEstimate(self):
		cardLogits=self.getLogitsOpponent
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.nnetsData["estimTarget"],logits=cardLogits))

	@define_scope
	def trainEstimate(self):
		optimizer=tf.train.AdamOptimizer(0.0001)
		variables = self.gModel.trainable_weights  #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "estimate_opponent_scope")
		return optimizer.minimize(self.costEstimate,var_list = variables)

	@define_scope
	def getLogitsValue(self):
		
		input_size = int(self.gameParams["inputSize"]+self.gameParams["handSize"])
		policy_size= int(self.gameParams["actionSize"])
		value_size= int(self.gameParams["valueSize"])
		
		inputs = Input(shape=(input_size,))
		model = Dense(output_dim=256, activation='relu')(inputs)
		for i in range(self.layers_f):
			model_prev = model
			model = Dense(output_dim=256, activation='relu')(model)
		logits = Dense(output_dim=policy_size, activation='linear')(model)
		v_values = Dense(output_dim=value_size, activation='linear')(model)
		self.fModel = Model(input=inputs, output=[logits , v_values])
		#print(self.getEstimateOpponent.get_shape())
		#print(self.nnetsData["input"].get_shape())
		return self.fModel(tf.concat(values=[self.nnetsData["input"],self.getLogitsOpponent],axis=1))

	@define_scope
	def getPolicyValue(self):
		logits ,v = self.getLogitsValue
		p = tf.nn.softmax(logits)
		v_reg = tf.nn.relu(v)
		return p,v_reg

	@define_scope
	def costPolicyValue(self):
		logits , v = self.getLogitsValue
		p_cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.nnetsData["policyTarget"],logits=logits))
		#print(p_cost)
		v_cost= tf.reduce_mean(tf.square(tf.subtract(v,self.nnetsData["valuesTarget"])))
		#print(v_cost)
		return tf.add(p_cost,tf.multiply(self.alpha,v_cost))

	@define_scope
	def trainPolicyValue(self):
		optimizer=tf.train.AdamOptimizer(0.0001)
		variables = self.fModel.trainable_weights#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "policy_value_scope")
		return optimizer.minimize(self.costPolicyValue, var_list=variables)

#Auxiliary functions
	def setAlpha(self,new_alpha):
		self.alpha=float(new_alpha)

	def policyValue(self,playerCard, publicHistory, publicCard):
		#print(publicCard)
		#print(" " + str(publicHistory.shape)+" " + str(playerCard.shape)+" "+str(publicCard.shape))
		playerInfo=self.preprocessInput(playerCard, publicHistory, publicCard)
		#print(playerInfo.shape)
		p, v = self.sess.run(self.getPolicyValue, feed_dict = {self.nnetsData["input"] : [playerInfo]})
		p=np.reshape(p,(self.gameParams["actionSize"]))
		v=np.reshape(v,(self.gameParams["valueSize"]))
		return p,v

	def estimateOpponent(self,playerCard, publicHistory, publicCard):
		playerInfo=self.preprocessInput(playerCard, publicHistory, publicCard)
		#print(playerInfo.shape)
		estimate=self.getEstimateOpponent.eval(session = self.sess, feed_dict = {self.nnetsData["input"]: [playerInfo]})

		return np.reshape(estimate,(self.gameParams["handSize"]))

	def preprocessInput(self, playerCard, publicHistory, publicCard): #Method that is here only because of the input specifics
		playerCard=np.reshape(playerCard,-1)
		publicHistory=np.reshape(publicHistory,-1)
		publicCard=np.reshape(publicCard,-1)
		return np.concatenate((playerCard,publicHistory,publicCard),axis=0)

	def initialiseIterator(self, reservoirs, miniBatchSize):
		feed_dict = {}
		for key,reservoir in reservoirs.items():
			feed_dict[self.rawData[key]] = reservoir
		feed_dict[self.batchSize] = miniBatchSize
		self.sess.run(self.iterator.initializer,feed_dict = feed_dict)

	def trainOnMinibatch(self):
		self.sess.run([self.trainEstimate,self.trainPolicyValue])

