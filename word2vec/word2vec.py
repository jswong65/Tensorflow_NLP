import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 20000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

def word2vec(batch_gen):
    """ Build the graph for word2vec model and train it """

    # Create a placeholder for input data
    inputs = tf.placeholder(tf.int32, 
                            shape=[None], 
                            name="inputs")

    # Create a placeholder for targets
    labels = tf.placeholder(tf.int32,
                            shape=[None, 1],
                            name="labels")    

    # Create a embedding matrix
    embeddings = tf.get_variable(name="embedings", 
                                shape=(VOCAB_SIZE, EMBED_SIZE),
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-1, 1))



    # Step 3: define the inference
    # get the embed of input words using tf.nn.embedding_lookup
    # embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

    # TO DO
    embed_words = tf.nn.embedding_lookup(embeddings, inputs)

    # create weights for noise-contrastive estimation (NCE)
    

        # Step 4: construct variables for NCE loss
        # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
        # nce_weight (vocab size x embed size), intialized to truncated_normal stddev=1.0 / (EMBED_SIZE ** 0.5)
        # bias: vocab size, initialized to 0

        # TO DO
    with tf.variable_scope("loss"):

        nce_weights = tf.get_variable(name="nce_weights",
                                    shape=(VOCAB_SIZE, EMBED_SIZE),
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))

        # create bias for noise-contrastive estimation (NCE)
        nce_biases = tf.get_variable(name="nce_biases",
                                    shape=[VOCAB_SIZE],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                                    

        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                            biases=nce_biases,
                            labels=labels,
                            inputs=embed_words,
                            num_sampled=NUM_SAMPLED,
                            num_classes=VOCAB_SIZE,
                            name='nce_loss'))

        # define loss function to be NCE loss function
        # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
        # need to get the mean accross the batch
        # note: you should use embedding of center words for inputs, not center words themselves

        # TO DO

        
    # Step 5: define optimizer
    
    # TO DO

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        # TO DO: initialize variables
        sess.run(tf.global_variables_initializer())

        total_loss = 0.0 # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./graphs/no_frills/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            # TO DO: create feed_dict, run optimizer, fetch loss_batch
            loss_batch, _ = sess.run([loss, optimizer], 
                                    feed_dict={inputs:centers, labels:targets})

            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()