class ModelConfig(object):
	def __init__(self):
		# configuration for building a model
		self.vocab_size = 5000
		self.batch_size = 32
		self.seq_size = 30
		self.image_size = 4096
		self.word_embed_size = 300
		self.image_embedding_size = 256
		self.lstm_size = 256
		self.num_layers = 1
		self.keep_prob = 0.7
    
    	# configuration for training
		self.learning_rate = 0.001
		self.grad_clip = 5
		self.optimizer = 'Adam'