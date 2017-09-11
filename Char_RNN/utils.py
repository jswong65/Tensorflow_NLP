import numpy as np

def extract_char_vocab(data):
	vocab = sorted(set(data))
	vocab_to_int = {c: i for i, c in enumerate(vocab)}
	int_to_vocab = dict(enumerate(vocab))

	return vocab_to_int, int_to_vocab

def encode_data(data, vocab_to_int):
	encoded = np.array([vocab_to_int[ch] for ch in text], dtype=np.int32)
	return encoded


def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = n_seqs * n_steps
    n_batches = arr.size // characters_per_batch

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * characters_per_batch]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, n_steps * n_batches))
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y