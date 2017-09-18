import numpy as np
import tensorflow as tf

class SkipGramModel:
	def __init__(self, params):
		# intialize the model parameters and variables
		self.params = params
		self.inputs = None
		self.labels = None
		self.embeddings = None



	def _build_placeholders(self):
		"""
			build placeholders for both input data and labels
		"""

		# Create a placeholder for input data
	    self.inputs = tf.placeholder(tf.int32, 
	                            shape=[None], 
	                            name="inputs")

	    # Create a placeholder for targets
	    self.labels = tf.placeholder(tf.int32,
	                            shape=[None, 1],
                            	name="labels")    

	def _build_embedding(self):
		"""
			build word embeddin matrix and 
		"""

		with tf.variable_scope("embedding")
			# Create the embedding matrix
			embeddings = tf.get_variable(name="embedings", 
	                                shape=(self.params.VOCAB_SIZE, self.params.EMBED_SIZE),
	                                dtype=tf.float32,
	                                initializer=tf.random_uniform_initializer(-1, 1))


	def _build_loss(self):
		"""
			Calculate and obtain the loss function with noise-contrastive estimation 
		"""

		with tf.variable_scope("loss"):
			# Use embedding_lookup to map inputs to word embeddings 
		    embed_words = tf.nn.embedding_lookup(self.embeddings, self.inputs)

			# create weights for noise-contrastive estimation (NCE)
	        nce_weights = tf.get_variable(name="nce_weights",
	                                    shape=(self.params.VOCAB_SIZE, self.params.EMBED_SIZE),
	                                    dtype=tf.float32,
	                                    initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))

	        # create biases for noise-contrastive estimation (NCE)
	        nce_biases = tf.get_variable(name="nce_biases",
	                                    shape=[self.params.VOCAB_SIZE],
	                                    dtype=tf.float32,
	                                    initializer=tf.zeros_initializer())
	                                    

	        self.loss = tf.reduce_mean(
	                tf.nn.nce_loss(weights=nce_weights,
	                            biases=nce_biases,
	                            labels=self.labels,
	                            inputs=embed_words,
	                            num_sampled=self.params.NUM_SAMPLED,
	                            num_classes=self.params.VOCAB_SIZE,
	                            name='nce_loss'))

	def _build_optimizer(self):
		"""
			build the optimizer using gradient descent
		"""
		self.optimizer = tf.train.GradientDescentOptimizer(self.params.LEARNING_RATE).minimize(self.loss)