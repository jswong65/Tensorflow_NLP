from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model import SkipGramModel
from config import ModelConfig
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

def word2vec():
	FLAGS = ModelConfig()
	model = SkipGramModel(FLAGS)
	model._build_model()
	batch_gen = process_data(FLAGS.VOCAB_SIZE, 
							FLAGS.BATCH_SIZE, 
							FLAGS.SKIP_WINDOW)

	# create a saver object
	saver = tf.train.Saver()

	# launch a session to compute the graph
	with tf.Session() as sess:
		# initialize all of the variables in the graph
		sess.run(tf.global_variables_initializer())

		total_loss = 0.0 # we use this to calculate the average loss in the last SKIP_STEP steps
		writer = tf.summary.FileWriter('./graphs/no_frills/', sess.graph)
		for index in range(FLAGS.NUM_TRAIN_STEPS):
			centers, targets = next(batch_gen)
			feed_dict = {model.inputs:centers, model.labels:targets}
			# excute the loss and optimizer op
			loss_batch, _ summary = sess.run([model.loss, model.optimizer,
									model.summary_op], 
									feed_dict=feed_dict)

			writer.add_summary(summary, global_step=model.global_step)
			total_loss += loss_batch
			if (index + 1) % FLAGS.SKIP_STEP == 0:
				saver.save(sess, 'checkpoints/skip-gram', global_step=model.global_step)
				print('Average loss at step {}: {:5.1f}'.format(index, total_loss / FLAGS.SKIP_STEP))
				total_loss = 0.0


		writer.close()


def main():
	word2vec()

if __name__ == '__main__':
	main()
