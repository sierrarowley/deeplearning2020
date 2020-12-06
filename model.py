import numpy as np
import tensorflow as tf
from preprocess import get_data
from embedding import glove_data, train_embedding, glove_embedding


def build_model(embedding_layer):
    """Since the output of the AD Assessment Engine was activated by a sigmoid function, it ranges from 0 to 1 and could be treated as a probability.
    The corresponding label for each output was thus 0 for subjects without AD and 1 for subjects with AD. The loss function was defined as the cross
    entropy sum between the out- put and the label of all training samples in a batch.
    BPTT is carried out using Adam with a learning rate of 0.001 as the optimizer. The batch size is set to 16 throughout the whole training process.
    All weights are initialized by using the Glorot normal initializer."""

    model = tf.keras.models.Sequential()

    # if given a pretrained embedding
    if embedding_layer:
        model.add(embedding_layer)
    # else use new embedding
    else:
        model.add(tf.keras.layers.Embedding(input_dim=1701, output_dim=256, input_length=552))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, activation="tanh")))
    model.add(tf.keras.layers.Dense(512,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.15, noise_shape=None, seed=None))
    model.add(tf.keras.layers.Dense(256,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.15, noise_shape=None, seed=None))
    model.add(tf.keras.layers.Dense(128,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Softmax())
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return model


def ten_sample():
    """ This method runs the model with cross validation. Ten samples will be used for testing and
    the rest for training each round. This is an alternative to how the model is run in main().
    """
    f = open("data.txt", "a")
    for i in range(0, 10):
        print(str(i))
        f.write("Run :" + str(i) + "\n")
        data_ids, labels, vocab_dict, cons_ids, cons_labels = get_data()
        labels=tf.one_hot(labels, 2)
        train_ids = data_ids[:446, :]
        train_labels = labels[:446]
        test_ids = data_ids[446:, :]
        test_labels = labels[446:]

        # training embeddings on only control data
        embedding_layer = train_embedding(cons_ids, cons_labels)

        #training embeddings on GLOVE embeddings
        embedding_index = glove_data()
        glove_layer = glove_embedding(vocab_dict, embedding_index)

        model = build_model(glove_layer)

        train_results = model.fit(train_ids, train_labels, shuffle=True, epochs=10, batch_size=100)
        loss_history = train_results.history["loss"]
        comma = ", "
        print(loss_history)
        print(comma.join(map(str, loss_history)))
        acc_history = train_results.history["categorical_accuracy"]

        f.write("\t Losses: ")
        f.write(comma.join(map(str,loss_history)))
        f.write("\n\t Accuracies: ")
        f.write(comma.join(map(str,acc_history)))
        test_results = model.evaluate(test_ids, test_labels, batch_size=16)
        f.write("\n\t Test Results: ")
        f.write(comma.join(map(str,test_results)))
        f.write("\n")

    f.close()


def main():
    # Use ten_sample() instead of the rest of main to perform cross validation
    # ten_sample()

    # otherwise this will just use 85-15 train-test split
    data_ids, labels, vocab_dict, cons_ids, cons_labels = get_data()
    labels=tf.one_hot(labels, 2)

    # 85% is training data, 15% is test data
    train_ids = data_ids[:446, :]
    train_labels = labels[:446]
    test_ids = data_ids[446:, :]
    test_labels = labels[446:]

    # these are used for the validation set
    train_val = train_ids[-70:]
    label_val = train_labels[-70:]

    # first train embeddings on control data
    embedding_layer = train_embedding(cons_ids, cons_labels)
    # OR train embeddings on GLOVE data
    embedding_index = glove_data()
    glove_layer = glove_embedding(vocab_dict, embedding_index)

    # build the model
    model = build_model(None)

    # train the model for 10 epochs
    train_results = model.fit(train_ids, train_labels, shuffle=True, epochs=10, batch_size=100)

    #test model
    print("Evaluate on test data")
    test_results = model.evaluate(test_ids, test_labels, batch_size=16)
    print("test loss, test acc:", test_results)

if __name__ == "__main__":
    main()
