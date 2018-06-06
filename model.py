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

	def __init__(self,session, gameParams,alpha =10.,feedGIntoF = False, batchSize = 128,hyp=None, poker = "leduc"):
		self.sess=session
		self.gameParams=gameParams
		self.poker = poker
		self.feedGIntoF = feedGIntoF
		self.predictionInput = tf.placeholder(dtype=tf.float32,shape=(None,gameParams["inputSize"]))
		self.firstCard = tf.placeholder(dtype=tf.float32,shape=(None,gameParams["deckSize"]))

		#Create placeholders
		self.rawPData={ "input" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["inputSize"])),
						"policyTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["actionSize"])) }
		
		self.rawVData={ "input" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["inputSize"])),
						"estimTarget" : tf.placeholder(dtype=tf.int32,shape=(None,gameParams["handSize"])),
						"valuesTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["valueSize"])) }
		
		self.pDataset = tf.data.Dataset.from_tensor_slices(self.rawPData)
		self.pDataset = self.pDataset.batch(batchSize)
		self.pDataset = self.pDataset.prefetch(1)
		self.pIterator = self.pDataset.make_initializable_iterator()

		self.pnNetsData =self.pIterator.get_next(name = "pIterator")

		self.vDataset = tf.data.Dataset.from_tensor_slices(self.rawVData)
		self.vDataset = self.vDataset.shuffle(buffer_size=10000)
		self.vDataset = self.vDataset.batch(batchSize)
		self.vDataset = self.vDataset.prefetch(1)
		self.vIterator = self.vDataset.make_initializable_iterator()

		self.vnNetsData =self.vIterator.get_next(name = "vIterator")

		#Properties that we are setting to be constants
		if hyp != None:
			self.alpha=tf.constant(hyp["alpha"],dtype=tf.float32)		
			self.lmbda = tf.constant(hyp["lmbda"],dtype=tf.float32)
			self.gLearningRate=hyp["gLearn"]
			self.fLearningRate=hyp["fLearn"]
		else:
			self.alpha=tf.constant(alpha,dtype=tf.float32)		
			self.lmbda = tf.constant(0.01,dtype=tf.float32)
			self.gLearningRate=0.0005
			self.fLearningRate=0.0005


		#Properties that are actually graph nodes
		self.gModel
		self.fModel
		self.valueLayer
		self.policyLayer
		self.costPolicyValue
		self.trainPolicyValue
		self.costEstimate
		self.trainEstimate
		self.getPolicyValue #sets up the graph and computes the Policy/Value
		self.getEstimateOpponent

#Functions that set properties during initialization
	@define_scope
	def gModel(self):
		#Takes an input and returns logits for opponent card
		if self.poker == "holdem":
			input_size = int(self.gameParams["inputSize"]+self.gameParams["deckSize"])
		else:
			input_size = int(self.gameParams["inputSize"])
		target_size= int(self.gameParams["deckSize"])
		inputs = Input(shape=(input_size,))
		model = Dense(units=256, activation='relu')(inputs)
		if self.poker == "holdem":
			model = Dense(units=256, activation='relu')(model)
			model = Dense(units=256, activation='relu')(model)
		model = Dense(units=128, activation='relu')(model)
		cards = Dense(units=target_size, activation='linear')(model)
		gModel = Model(inputs=inputs, outputs=cards)

		return gModel

	@define_scope
	def fModel(self):
		if not self.feedGIntoF:
			input_size = int(self.gameParams["inputSize"])    #Use if not feeding gModel into fModel
		else:
			input_size = int(self.gameParams["inputSize"]+self.gameParams["deckSize"])    # Use if feeding gModel into fModel

		inputs = Input(shape=(input_size,))
		model = Dense(units=256, activation='relu')(inputs)
		if self.poker == "holdem":
			model = Dense(units=256, activation='relu')(model)
			model = Dense(units=256, activation='relu')(model)
		model = Dense(units=128, activation='relu')(model)

		fModel = Model(inputs=inputs, outputs=model)

		return fModel

	@define_scope
	def valueLayer(self):
		value_size= int(self.gameParams["valueSize"])
		return Dense(units = value_size, input_shape = self.fModel.output_shape, activation='linear')

	@define_scope
	def policyLayer(self):
		policy_size= int(self.gameParams["actionSize"])
		return Dense(units = policy_size, input_shape = self.fModel.output_shape, activation = 'linear')

	@define_scope
	def vInputData(self):
		return tf.concat(values= [self.vnNetsData["playerCard"],self.vnNetsData["publicData"]],axis= 1)

	@define_scope
	def costPolicyValue(self):
		if self.feedGIntoF:
			logits = self.policyLayer(tf.concat(values=[self.pnNetsData["input"],self.gModel[self.pnNetsData["input"]]],axis=1))
			v = self.valueLayer(self.fModel(tf.concat(values=[self.vnNetsData["input"],self.gModel[self.vnNetsData["input"]]],axis=1)))
		else:
			logits = self.policyLayer(self.fModel(self.pnNetsData["input"]))
			v = self.valueLayer(self.fModel(self.vnNetsData["input"]))

		p_cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pnNetsData["policyTarget"],logits=logits))
		
		
		v_cost= tf.reduce_mean(tf.square(tf.subtract(v,self.vnNetsData["valuesTarget"])))
		#print(v_cost)

		cost = tf.add(tf.divide(p_cost,self.alpha),v_cost)

		for layer in self.fModel.layers:
			if len(layer.get_weights()) > 0:
				cost = tf.add(cost,tf.multiply(self.lmbda,tf.nn.l2_loss(layer.get_weights()[0])))

		cost = tf.add(cost,tf.multiply(self.lmbda,tf.nn.l2_loss(self.policyLayer.get_weights()[0])))
		cost = tf.add(cost,tf.multiply(self.lmbda,tf.nn.l2_loss(self.valueLayer.get_weights()[0])))

		return cost


	@define_scope
	def trainPolicyValue(self):
		optimizer=tf.train.AdamOptimizer(self.fLearningRate)
		variables = self.fModel.trainable_weights
		variables.append(self.valueLayer.trainable_weights)
		variables.append(self.policyLayer.trainable_weights)
		return optimizer.minimize(self.costPolicyValue, var_list=variables)

	@define_scope
	def costEstimate(self):
		inputs = self.vnNetsData["input"]

		if self.poker == "holdem":
			inputs = tf.pad(inputs,tf.constant([[0,0],[0,self.Params["deckSize"]]]))

		cardLogits=self.gModel(inputs)
		oneHot = tf.one_hot(self.vnNetsData["estimTarget"],self.gameParams["deckSize"])
		tf.Assert(tf.equal(tf.rank(oneHot), tf.constant(3)),[oneHot])
		labels = tf.reduce_mean(oneHot , axis = 1)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=cardLogits))

		if self.poker == "holdem":
			reshapedVnnetsData = tf.expand_dims(self.vnNetsData["input"],1)+tf.zeros((self.vnNetsData["input"].shape[0],2,self.vnNetsData["input"].shape[1]))
			inputs = tf.reshape(tf.concat(values =[reshapedVnnetsData,oneHot],axis = 2),shape = [-1,self.gameParams["inputSize"]+self.gameParams["deckSize"]])
			cardLogits=self.gModel(inputs)
			labels = tf.reshape(tf.reverse(oneHot, axis = 1), shape = [-1,oneHot.shape[2]])
			cost = tf.add(cost,tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=cardLogits)))

		for layer in self.gModel.layers:
			if len(layer.get_weights()) > 0:
				cost = tf.add(cost,tf.multiply(self.lmbda,tf.nn.l2_loss(layer.get_weights()[0])))
		return cost

	@define_scope
	def trainEstimate(self):
		optimizer=tf.train.AdamOptimizer(self.gLearningRate)
		variables = self.gModel.trainable_weights 
		return optimizer.minimize(self.costEstimate,var_list = variables)

	@define_scope
	def getPolicyValue(self):
		#Used for predictions only
		if self.feedGIntoF:
			lastHiddenLayer = self.fModel(tf.concat(values=[self.predictionInput,self.gModel[self.predictionInput]],axis=1))
		else:
			lastHiddenLayer = self.fModel(self.predictionInput)

		p = tf.nn.softmax(self.policyLayer(lastHiddenLayer))
		v = tf.nn.relu(self.valueLayer(lastHiddenLayer))
		return p,v

	@define_scope
	def getEstimateOpponent(self):
		#Used for predictions only
		if self.poker == "holdem":
			return tf.nn.softmax(self.gModel(tf.concat([self.predictionInput,self.firstCard],axis = 1)))
		else:
			return tf.nn.softmax(self.gModel(self.predictionInput))

#Auxiliary functions

	def compute_cost_alpha(self):

		cost = self.sess.run(self.costPolicyValue)

		return cost

	def setAlpha(self,new_alpha):
		self.alpha=float(new_alpha)

	def policyValue(self,playerCard, publicHistory, publicCard):
		#print(publicCard)
		#print(" " + str(publicHistory.shape)+" " + str(playerCard.shape)+" "+str(publicCard.shape))
		playerInfo=self.preprocessInput(playerCard, publicHistory, publicCard)
		#print(playerInfo.shape)
		p, v = self.sess.run(self.getPolicyValue, feed_dict = {self.predictionInput : [playerInfo] })
		p=np.reshape(p,(self.gameParams["actionSize"]))
		v=np.reshape(v,(self.gameParams["valueSize"]))
		return p,v

	def estimateOpponent(self, playerCard, publicHistory, publicCard, firstCard = None):
		playerInfo=self.preprocessInput( playerCard, publicHistory, publicCard)
		#print(playerInfo.shape)
		if self.poker == "holdem":
			if firstCard is None:
				firstCard = np.zeros(self.gameParams["deckSize"])
			estimate=self.getEstimateOpponent.eval(session = self.sess, feed_dict = {self.predictionInput: [playerInfo], self.firstCard : [firstCard]})
		else:
			estimate=self.getEstimateOpponent.eval(session = self.sess, feed_dict = {self.predictionInput: [playerInfo]})

		return np.reshape(estimate,(self.gameParams["deckSize"]))

	def preprocessInput(self, playerCard, publicHistory, publicCard): #Method that is here only because of the input specifics
		playerCard=np.reshape(playerCard,-1)
		publicHistory=np.reshape(publicHistory,-1)
		publicCard=np.reshape(publicCard,-1)
		return np.concatenate((playerCard,publicHistory,publicCard),axis=0)

	def initialisePIterator(self, pReservoirs):
		feed_dict = {}
		for key,reservoir in pReservoirs.items():
			feed_dict[self.rawPData[key]] = reservoir
		self.sess.run(self.pIterator.initializer,feed_dict = feed_dict)

	def initialiseVIterator(self, vReservoirs):
		feed_dict = {}
		for key,reservoir in vReservoirs.items():
			feed_dict[self.rawVData[key]] = reservoir
		self.sess.run(self.vIterator.initializer,feed_dict = feed_dict)



	def trainOnMinibatch(self):
		try:
			self.sess.run([self.trainEstimate,self.trainPolicyValue])
		except tf.errors.OutOfRangeError as error:
			return error.op.name
		except  tf.errors.FailedPreconditionError as error:
			return error.op.name
		return None