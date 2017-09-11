import time
import model
import config
import tensorflow as tf

from utils import extract_char_vocab, encode_data, get_batches

config = config.ModelConfig()
model = model.CharRNNMoel(config, )
model.build_model()
saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
    
	# Use the line below to load a checkpoint and resume training
	#saver.restore(sess, 'checkpoints/______.ckpt')
	counter = 0
	for e in range(epochs):
    	# Train network
    	new_state = sess.run(model.initial_state)
    	loss = 0
    	for x, y in get_batches(encoded, batch_size, num_steps):
    		counter += 1
    		start = time.time()
    		feed = {model.inputs: x,
    				model.targets: y,
    				model.keep_prob: config.keep_prob,
    				model.initial_state: new_state}
    		
    		batch_loss, new_state, _ = sess.run([model.loss, 
    											model.final_state, 
    											model.optimizer], 
    											feed_dict=feed)
            
			end = time.time()
			print('Epoch: {}/{}... '.format(e+1, epochs),
				'Training Step: {}... '.format(counter),
				'Training loss: {:.4f}... '.format(batch_loss),
				'{:.4f} sec/batch'.format((end-start)))
        
			if (counter % save_every_n == 0):
				saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    
	saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))