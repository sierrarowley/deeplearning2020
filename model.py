import numpy as np
import tensorflow as tf
from preprocess import get_data


def main():
    data_ids, labels, vocab_dict = get_data()

    # 85% is training data, 15% is test data
    train_ids = data_ids[:470, :]
    train_labels = labels[:470]
    test_ids = data_ids[470:, :]
    test_labels = labels[470:]

if __name__ == "__main__":
    main()
