import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist, fashion_mnist


def download_mnist_datasets(datatype):
    # MNISTのデータダウンロード（tensorflow用）
    if datatype == "mnist":
        data = input_data.read_data_sets("datasets/mnist", one_hot=True, source_url=None)
    elif datatype == "fashion_mnist":
        data = input_data.read_data_sets("datasets/fashion_mnist", one_hot=True, source_url=None)
    else:
        raise Exception
    return data


def read_mnist_datasests(datatype, batch_size, option):
    if option == "train":
        x, y = datatype.train.next_batch(batch_size)
    elif option == "test":
        x, y = datatype.test.next_batch(batch_size)
    else:
        raise Exception

    x_data = []

    for data in x:
        x_data.append(np.reshape(data, (28, 28, 1)))
    x_data = np.array(x_data)

    return x_data, y


def download_mnist_for_keras():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    pass