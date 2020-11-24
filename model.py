import numpy as np
import tensorflow as tf
from preprocess import get_data


def main():
    data_ids, labels, vocab_dict = get_data()
    print(data_ids)
    print(labels)


if __name__ == "__main__":
    main()
