import numpy as np
import tensorflow as tf
from preprocess import get_data

from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers.recurrent import GRU


def build_model():
<<<<<<< HEAD
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 128)
    model.add(Bidirectional(layers.GRU(128, return_sequences=True)))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
=======
    """Since the output of the AD Assessment Engine was activated by a sigmoid function, it ranges from 0 to 1 and could be treated as a probability. 
    The corresponding label for each output was thus 0 for subjects without AD and 1 for subjects with AD. The loss function was defined as the cross 
    entropy sum between the out- put and the label of all training samples in a batch. 
    BPTT is carried out using Adam with a learning rate of 0.001 as the optimizer. The batch size is set to 16 throughout the whole training process. 
    All weights are initialized by using the Glorot normal initializer."""


    # https://www.tensorflow.org/tutorials/text/text_classification_rnn <--- should we try this instead?
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=1781, output_dim=128, input_length=635)) #output should be (batch_size, input length, output_dim). number of words cannot exceed 1780
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 128)
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, activation="relu", kernel_initializer='glorot_normal', recurrent_activation="sigmoid", return_sequences=True)))
    model.add(tf.keras.layers.Dense(64, use_bias=True,))
    model.add(tf.keras.layers.Dense(1, use_bias=True,))
    model.add(tf.keras.layers.Activation('sigmoid'))
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM,), #https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001,),
        metrics=[tf.keras.metrics.BinaryAccuracy()],)
    model.summary()
>>>>>>> f0f7a82... ushas tuning
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
<<<<<<< HEAD
    train_ids=data_ids[:470, :]
    train_labels=labels[:470]
    test_ids=data_ids[470:, :]
    test_labels=labels[470:]
=======
    train_ids = data_ids[:470, :]
    train_labels = labels[:470]
    test_ids = data_ids[470:, :]
    test_labels = labels[470:]

    print(train_ids.shape)
    train_val = train_ids[-70:]
    label_val = train_labels[-70:]
    # train_ids = train_ids[:-70]
    # train_labels = train_labels[:-70]
>>>>>>> f0f7a82... ushas tuning

    model = build_model()
<<<<<<< HEAD
    
=======

    # train the model for 2 epochs
    # train_results = model.fit(train_ids, train_labels, validation_data=(train_val, label_val), epochs=10, batch_size=16)
    train_results = model.fit(train_ids, train_labels, shuffle=True, epochs=10, batch_size=64)

    # model.train(model, train_ids, train_labels)

    #test model
    print("Evaluate on test data")
    test_results = model.evaluate(test_ids, test_labels, batch_size=64)
    print("test loss, test acc:", test_results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(test_ids[:3])
    print("predictions shape:", predictions.shape)
>>>>>>> f0f7a82... ushas tuning

if __name__ == "__main__":
    main()
