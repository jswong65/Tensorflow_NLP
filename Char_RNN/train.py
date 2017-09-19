import time
from model import CharRNNMoel
from config import ModelConfig

import tensorflow as tf
import numpy as np

from utils import extract_char_vocab, encode_data, get_batches


def train():

	FLAGS = ModelConfig()
	model = CharRNNMoel(FLAGS, "train")
	model.build_model()
	saver = tf.train.Saver(max_to_keep=100)

	with open('anna.txt', 'r') as f:
	    text=f.read()
	vocab = sorted(set(text))
	vocab_to_int = {c: i for i, c in enumerate(vocab)}
	int_to_vocab = dict(enumerate(vocab))
	encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	    
		# Use the line below to load a checkpoint and resume training
		#saver.restore(sess, 'checkpoints/______.ckpt')
		counter = 0
		for e in range(FLAGS.epochs):
			# Train network
			new_state = sess.run(model.initial_state)
			loss = 0
			for x, y in get_batches(encoded, FLAGS.batch_size, FLAGS.num_steps):
				counter += 1
				start = time.time()
				feed = {model.inputs: x,
	    				model.targets: y,
	    				model.keep_prob: FLAGS.keep_prob,
	    				model.initial_state: new_state}
	    		
				batch_loss, new_state, _ = sess.run([model.loss, 
	    											model.final_state, 
	    											model.train_op], 
	    											feed_dict=feed)
	            
				end = time.time()
				print('Epoch: {}/{}... '.format(e+1, FLAGS.epochs),
					'Training Step: {}... '.format(counter),
					'Training loss: {:.4f}... '.format(batch_loss),
					'{:.4f} sec/batch'.format((end-start)))
	        
				if (counter % FLAGS.save_every_n == 0):
					saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, FLAGS.lstm_size))
	    
		saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, FLAGS.lstm_size))

def main():
	train()

if __name__ == "__main__":
	main()