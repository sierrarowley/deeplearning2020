import numpy as np
import tensorflow as tf

def train_embedding(train_vals, train_labels):
    """ This method trains embeddings on the DementiaBank control data. This allows the embeddings
    to learn from unbroken speech, which might occur in the dementia patient data. These embeddings
    can be fed into the model instead of using new embeddings.

    :param train_vals: the control data ids
    :param train_labels: the control data labels
    :return: an embedding layer
    """
    model = tf.keras.Sequential()
    embedding_layer = tf.keras.layers.Embedding(input_dim=1701, output_dim=256, input_length=552)
    model.add(embedding_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.fit(train_vals, train_labels, shuffle=True, epochs=8, batch_size=64)
    return embedding_layer


def glove_embedding(vocab_dict, embeddings_index):
    """ This method trains embeddings on the glove data. This allows the embeddings to learn from
    unbroken speech, which might occur in the dementia patient data. These embeddings can be fed
    into the model instead of using new embeddings.

    :param vocab_dict: the DementiaBank data vocab dict
    :param embeddings_index: the glove data vocab dict
    :return: an embedding layer
    """
    misses = 0
    embedding_matrix = np.zeros((len(vocab_dict) + 1, 100))
    for word, i in vocab_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            misses +=1

    embedding_layer = tf.keras.layers.Embedding(input_dim=1841, output_dim=100, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), input_length=552, trainable='false')
    print('misses', misses)
    return embedding_layer


def glove_data():
    """ This method reads in the glove data.
    """
    embeddings_index = {}
    f = open('glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index
