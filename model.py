import numpy as np
import tensorflow as tf
from preprocess import get_data

from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers.recurrent import GRU


def build_model():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 128)
    model.add(Bidirectional(layers.GRU(128, return_sequences=True)))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train(model, train_ids, train_labels)):
    """
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_ids:  train data (all data for training) of shape (num_sentences, maxlen)
	:param train_labels: train labels (all data for training) of shape (num_sentences, 1); label is 0 if control, 1 if dementia patient
	:return: None
	"""
    # can mask some words if results are not goood enough
    pass

def test(model, test_ids, test_labels):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
	:param test_ids:  train data (all data for training) of shape (num_sentences, maxlen)
	:param test_labels: train labels (all data for training) of shape (num_sentences, 1); label is 0 if control, 1 if dementia patient
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
    e.g. (my_perplexity, my_accuracy)
    """
    pass

def main():
    data_ids, labels, vocab_dict=get_data()

    # 85% is training data, 15% is test data
    train_ids=data_ids[:470, :]
    train_labels=labels[:470]
    test_ids=data_ids[470:, :]
    test_labels=labels[470:]

    model = build_model()
    

if __name__ == "__main__":
    main()
