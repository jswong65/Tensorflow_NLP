class ModelConfig(object):
	def __init__(self):
		# configuration for building a model
		self.batch_size = 64
		self.num_steps = 50
		self.vocab_size = 83
		self.lstm_size = 128
		self.num_layers = 2
		self.keep_prob = 0.5
    	
    	# configuration for training
		self.learning_rate = 0.001
		self.grad_clip = 5
		self.epochs = 5
		self.save_every_n = 200