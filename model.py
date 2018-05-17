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
	def __init__(self,session, gameParams,alpha =1.):
		session.run(A)
		self.sess=session

		#Create placeholders
		self.policyNetTarget=tf.placeholder(dtype=tf.float32,shape=(None,gameParams["actionSize"]))
		self.valueNetTarget=tf.placeholder(dtype=tf.float32,shape=(None,gameParams["valueSize"]))
		self.nnetsInput=tf.placeholder(dtype=tf.float32,shape=(None,gameParams["historySize"]+gameParams["handSize"]+gameParams["publicCardSize"]))
		self.estimNetTarget=tf.placeholder(dtype=tf.float32,shape=(None,gameParams["handSize"]))

		#Properties that we are setting to be constants
		self.alpha=float(alpha)

		#Model properties
		self.gModel=None
		self.fModel=None

		#Properties that are actually graph nodes
		self.getEstimateOpponent
		self.costEstimate
		self.trainEstimate
		self.getPolicyValue #sets up the graph and computes the Policy/Value
		self.costPolicyValue
		self.trainPolicyValue

#Functions that set properties during initialization
	@define_scope
	def getEstimateOpponent(self):
		with tf.variable_scope("estimate_opponent_scope"):
			print("something")
			input_size = int(self.nnetsInput.get_shape()[1])
			target_size= int(self.estimNetTarget.get_shape()[1])
			inputs = Input(shape=(input_size,))
			model = Dense(output_dim=256, activation='relu')(inputs)
			model = Dense(output_dim=256, activation='relu')(model)
			cards = Dense(output_dim=target_size, activation='softmax')(model)
			self.gModel = Model(input=inputs, output=cards)
		return self.gModel(self.nnetsInput)

	@define_scope
	def costEstimate(self):
		cardProb=self.getEstimateOpponent
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.estimNetTarget,logits=cardProb))

	@define_scope
	def trainEstimate(self):
		optimizer=tf.train.AdamOptimizer(0.0001)
		variables = self.gModel.trainable_weights  #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = "estimate_opponent_scope")
		return optimizer.minimize(self.costEstimate,var_list = variables)

	@define_scope
	def getPolicyValue(self):
		with tf.variable_scope("policy_value_scope"):
			input_size = int(self.nnetsInput.get_shape()[1]+self.estimNetTarget.get_shape()[1])
			policy_size= int(self.policyNetTarget.get_shape()[1])
			value_size= int(self.valueNetTarget.get_shape()[1])
			inputs = Input(shape=(input_size,))
			model = Dense(output_dim=256, activation='relu')(inputs)
			model = Dense(output_dim=256, activation='relu')(model)
			p_values = Dense(output_dim=policy_size, activation='softmax')(model)
			v_values = Dense(output_dim=value_size, activation='linear')(model)
			self.fModel = Model(input=inputs, output=[p_values , v_values])
			print(self.getEstimateOpponent.get_shape())
			print(self.nnetsInput.get_shape())
		return self.fModel(tf.concat(values=[self.nnetsInput,self.getEstimateOpponent],concat_dim = 1))

	@define_scope
	def costPolicyValue(self):
		p , v = self.getPolicyValue
		p_cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.policyNetTarget,logits=p))
		print(p_cost)
		v_cost= tf.reduce_mean(tf.square(tf.subtract(v,self.valueNetTarget)))
		print(v_cost)
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

		playerInfo=self.preprocessInput(playerCard, publicHistory, publicCard)
		return self.sess.run(self.getPolicyValue, feed_dict = {self.nnetsInput : [playerInfo]})

	def estimateOpponent(self,playerCard, publicHistory, publicCard,flattened=False):

		playerInfo=self.preprocessInput(playerCard, publicHistory, publicCard)

		return self.getEstimateOpponent.eval(session = self.sess, feed_dict = {self.nnetsInput: [playerInfo]})

	def preprocessInput(self, playerCard, publicHistory, publicCard): #Method that is here only because of the input specifics
		playerCard=np.reshape(playerCard,-1)
		publicHistory=np.reshape(publicHistory,-1)
		publicCard=np.reshape(publicCard,-1)
		return np.concatenate((playerCard,publicHistory,publicCard),axis=0)

	def trainNets(self, minibatchTuple):
		inputs, policies, values, oppCards = minibatchTuple
		self.sess.run([trainEstimate,trainPolicyValue],feed_dict = {self.nnetsInput:[inputs],self.policyNetTarget:[policies],self.valueNetTarget:values, self.estimNetTarget: oppCards} )