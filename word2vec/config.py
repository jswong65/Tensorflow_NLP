class ModelConfig(object):
	def __init__(self):
		# configuration for building a model
		self.VOCAB_SIZE = 50000
		self.BATCH_SIZE = 128
		self.EMBED_SIZE = 128 # dimension of the word embedding vectors
		self.SKIP_WINDOW = 1 # the context window
		self.NUM_SAMPLED = 64    # Number of negative examples to sample.
		self.LEARNING_RATE = 1.0
		self.NUM_TRAIN_STEPS = 20000
		self.SKIP_STEP = 2000 # how many steps to skip before reporting the loss