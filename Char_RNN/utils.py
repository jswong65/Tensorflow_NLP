import numpy as np

def extract_char_vocab(data):
	vocab = sorted(set(data))
	vocab_to_int = {c: i for i, c in enumerate(vocab)}
	int_to_vocab = dict(enumerate(vocab))

	return vocab_to_int, int_to_vocab

def encode_data(data, vocab_to_int):
	encoded = np.array([vocab_to_int[ch] for ch in text], dtype=np.int32)