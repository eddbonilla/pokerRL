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
	def __init__(self,session, gameParams,alpha =40.,feedGIntoF = False, batchSize = 128, lmbda = 0.005):

		self.sess=session
		self.gameParams=gameParams
		self.feedGIntoF = feedGIntoF

		#Create placeholders
		self.predictionInput = tf.placeholder(dtype=tf.float32,shape=(None,gameParams["inputSize"]))
		self.predictionPublicData = tf.placeholder(dtype=tf.float32,shape=(None,gameParams["historySize"]+self.gameParams["handSize"]))
		self.rawPData={ "input" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["inputSize"])),
						"policyTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["actionSize"])) }
		
		self.rawVData={ "playerCard" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["handSize"])),
						"publicData" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["historySize"]+gameParams["handSize"])),
						"estimTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["handSize"])),
						"valuesTarget" : tf.placeholder(dtype=tf.float32,shape=(None,gameParams["valueSize"])) }
		
		self.pDataset = tf.data.Dataset.from_tensor_slices(self.rawPData)
		self.pDataset = self.pDataset.repeat()
		self.pDataset = self.pDataset.batch(batchSize)
		self.pDataset = self.pDataset.prefetch(1)
		self.pIterator = self.pDataset.make_initializable_iterator()

		self.pnNetsData =self.pIterator.get_next()

		self.vDataset = tf.data.Dataset.from_tensor_slices(self.rawVData)
		self.vDataset = self.vDataset.repeat()
		self.vDataset = self.vDataset.shuffle(buffer_size=16384)
		self.vDataset = self.vDataset.batch(batchSize)
		self.vDataset = self.vDataset.prefetch(1)
		self.vIterator = self.vDataset.make_initializable_iterator()

		self.vnNetsData =self.vIterator.get_next()

		#Properties that we are setting to be constants
		self.alpha=tf.constant(alpha)		
		self.lmbda = tf.constant(lmbda)

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
		input_size = int(self.gameParams["historySize"]+self.gameParams["handSize"])
		target_size= int(self.gameParams["handSize"])
		inputs = Input(shape=(input_size,))
		model = Dense(output_dim=256, activation='relu')(inputs)
		model = Dense(output_dim=128, activation='relu')(model)
		cards = Dense(output_dim=target_size, activation='linear')(model)
		gModel = Model(input=inputs, output=cards)

		return gModel

	@define_scope
	def fModel(self):
		if not self.feedGIntoF:
			input_size = int(self.gameParams["inputSize"])    #Use if not feeding gModel into fModel
		else:
			input_size = int(self.gameParams["inputSize"]+self.gameParams["handSize"])    # Use if feeding gModel into fModel

		inputs = Input(shape=(input_size,))
		model = Dense(output_dim=256, activation='relu')(inputs)
		model = Dense(output_dim=128, activation='relu')(model)

		fModel = Model(input=inputs, output=model)

		return fModel

	@define_scope
	def valueLayer(self):
		value_size= int(self.gameParams["valueSize"])
		return Dense(output_dim = value_size, input_shape = self.fModel.output_shape, activation='linear')

	@define_scope
	def policyLayer(self):
		policy_size= int(self.gameParams["actionSize"])
		return Dense(output_dim = policy_size, input_shape = self.fModel.output_shape, activation = 'linear')

	@define_scope
	def vInputData(self):
		return tf.concat(values= [self.vnNetsData["playerCard"],self.vnNetsData["publicData"]],axis= 1)

	@define_scope
	def costPolicyValue(self):
		if self.feedGIntoF:
			logits = self.policyLayer(tf.concat(values=[self.pnNetsData["input"],self.gModel[self.pnNetsData["input"][gameParams["handSize"]:]]],axis=1))
			v = self.valueLayer(self.fModel(tf.concat(values=[self.vInputData,self.gModel[self.vnNetsData["publicData"]]],axis=1)))

		else:
			logits = self.policyLayer(self.fModel(self.pnNetsData["input"]))
			v = self.valueLayer(self.fModel(self.vInputData))

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
		optimizer=tf.train.AdamOptimizer(0.0001)
		variables = self.fModel.trainable_weights
		variables.append(self.valueLayer.trainable_weights)
		variables.append(self.policyLayer.trainable_weights)
		return optimizer.minimize(self.costPolicyValue, var_list=variables)

	@define_scope
	def costEstimate(self):
		cardLogits=self.gModel(self.vnNetsData["publicData"])
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.vnNetsData["estimTarget"],logits=cardLogits))
		for layer in self.gModel.layers:
			if len(layer.get_weights()) > 0:
				cost = tf.add(cost,tf.multiply(self.lmbda,tf.nn.l2_loss(layer.get_weights()[0])))
		return cost

	@define_scope
	def trainEstimate(self):
		optimizer=tf.train.AdamOptimizer(0.0002)
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
		return tf.nn.softmax(self.gModel(self.predictionPublicData))


#Auxiliary functions
	def setAlpha(self,new_alpha):
		self.alpha=float(new_alpha)

	def policyValue(self,playerCard, publicHistory, publicCard):
		#print(publicCard)
		#print(" " + str(publicHistory.shape)+" " + str(playerCard.shape)+" "+str(publicCard.shape))
		playerInfo=self.preprocessInput(publicHistory, publicCard, playerCard= playerCard)
		#print(playerInfo.shape)
		p, v = self.sess.run(self.getPolicyValue, feed_dict = {self.predictionInput : [playerInfo] })
		p=np.reshape(p,(self.gameParams["actionSize"]))
		v=np.reshape(v,(self.gameParams["valueSize"]))
		return p,v

	def estimateOpponent(self, publicHistory, publicCard):
		playerInfo=self.preprocessInput(publicHistory, publicCard)
		#print(playerInfo.shape)
		estimate=self.getEstimateOpponent.eval(session = self.sess, feed_dict = {self.predictionPublicData: [playerInfo]})

		return np.reshape(estimate,(self.gameParams["handSize"]))

	def preprocessInput(self, publicHistory, publicCard, playerCard = None): #Method that is here only because of the input specifics
		publicHistory=np.reshape(publicHistory,-1)
		publicCard=np.reshape(publicCard,-1)
		if playerCard is not None:
			playerCard=np.reshape(playerCard,-1)
			concat = np.concatenate((playerCard,publicHistory,publicCard),axis=0)
		else:
			concat = np.concatenate((publicHistory,publicCard),axis=0)
		return concat

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
		self.sess.run([self.trainEstimate,self.trainPolicyValue])
