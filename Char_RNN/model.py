import tensorflow as tf

class CharRNNMoel(object):

	def __init__(self, config, mode):
		self.config = config
		self.mode = mode

		self.inputs = None
		self.targets = None
		self.keep_prob = None
		self.cell = None
		self.initial_state = None 
		self.logits = None
		self.outputs = None
		self.final_state = None

		if mode == "sampling":
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps



	def build_inputs(self):
		''' Define placeholders for inputs, targets, and dropout 
	    
	        Arguments
			---------
			batch_size: Batch size, number of sequences per batch
			num_steps: Number of sequence steps in a batch
	        
		'''
		# Declare placeholders we'll feed into the graph
		self.inputs = tf.placeholder(tf.int32, [self.config.batch_size, self.config.num_steps], name="inputs")
		self.targets = tf.placeholder(tf.int32, [self.config.batch_size, self.config.num_steps], name="targets")
	    
		# Keep probability placeholder for drop out layers
		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

	def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
		''' Build LSTM cell.
	    
			Arguments
			---------
			keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
			lstm_size: Size of the hidden layers in the LSTM cells
			num_layers: Number of LSTM layers
			batch_size: Batch size

		'''
		### Build the LSTM Cell
		def build_cell(num_units, keep_prob):
			# Use a basic LSTM cell
			lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
			# Add dropout to the cell outputs
			drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
			return drop
	    
		# Stack up multiple LSTM layers, for deep learning
		self.cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
		self.initial_state = cell.zero_state(batch_size, tf.float32)
	    


	def build_network(self):
		# Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        self.outputs, self.final_state = tf.nn.dynamic_rnn(cell,
                                           x_one_hot,
                                           initial_state=self.initial_state)

	def build_output(self):
		''' Build a softmax layer, return the softmax output and logits.
    
			Arguments
			---------
        
			lstm_output: List of output tensors from the LSTM layer
			in_size: Size of the input tensor, for example, size of the LSTM cells
			out_size: Size of this softmax layer
    
		'''

		# Reshape output so it's a bunch of rows, one row for each step for each sequence.
		# Concatenate lstm_output over axis 1 (the columns)
		seq_output = tf.concat(self.outputs, axis=1)
		# Reshape seq_output to a 2D tensor with lstm_size columns
		x = tf.reshape(seq_output, [-1, self.config.lstm_size])
    
		# Connect the RNN outputs to a softmax layer
		with tf.variable_scope('softmax'):
			# Create the weight and bias variables here
			softmax_w = tf.get_variable('softmax_w', 
										[self.config.lstm_size, self.config.vocab_size], 
										initializer=tf.truncated_normal_initializer(stddev=0.1))

			softmax_b = tf.get_variable('softmax_b', [self.config.vocab_size])
    
		# Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
		# of rows of logit outputs, one for each step and sequence
		self.logits = tf.matmul(x, softmax_w) + softmax_b
    
		# Use softmax to get the probabilities for predicted characters
		self.output = tf.nn.softmax(self.logits, name="predictions")
    
    def build_loss(self):
		''' Calculate the loss from the logits and the targets.
		
			Arguments
			---------
			logits: Logits from final fully connected layer
			targets: Targets for supervised learning
			lstm_size: Number of LSTM hidden units
			num_classes: Number of classes in targets
			
		'''
    
		# One-hot encode targets and reshape to match logits, one row per sequence per step
		y_one_hot = tf.one_hot(self.targets, self.config.vocab_size)
		y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape()) 
    
		# Softmax cross entropy loss
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
		self.loss = tf.reduce_mean(loss)
    
    def build_optimizer(self):
		''' Build optmizer for training, using gradient clipping.
		
			Arguments:
			loss: Network loss
			learning_rate: Learning rate for optimizer
			
		'''
    
		# Optimizer for training, using gradient clipping to control exploding gradients
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.grad_clip)
		train_op = tf.train.AdamOptimizer(self.config.learning_rate)
		self.optimizer = train_op.apply_gradients(zip(grads, tvars))

	def build_mode(self):
		self.build_inputs()
		self.build_lstm()
		self.build_output()
		self.build_loss()
		self.build_optimizer()