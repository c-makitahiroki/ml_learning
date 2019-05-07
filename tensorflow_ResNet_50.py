from utils.download_image import download_mnist_datasets, read_mnist_datasests
from nets.ResNet50 import resnet50
from train.train import exec_train
from eval.evaluation import exec_evaluation
from loss.loss import cross_entropy

import tensorflow as tf
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=str, default=128)
    parser.add_argument("--epoch_num", type=str, default=10)
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--datatype", type=str, default="fashion_mnist")
    parser.add_argument("--loss", dtype=str, default="cross_entropy")

    args = parser.parse_args()

    return args


def main(args):
    batch_size = args.batch_size
    epoch_num = args.epoch_num

    X = tf.placeholder("float", [batch_size, 28, 28, 1])
    Y = tf.placeholder("float", [batch_size, 10])
    learning_rate = tf.placeholder("float", [])

    # ネットワークの定義
    net = resnet50(X)

    # 損失関数と学習メソッドの定
    loss = cross_entropy(Y, net)
    train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

    """
    # Tensorboard
    with tf.name_scope('summary'):
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', sess.graph)
    """

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    data = download_mnist_datasets(args.datatype)

    # 学習
    sess = exec_train(train_op, X, Y, epoch_num, data, batch_size, loss, learning_rate, args.learning_rate)

    # 評価
    acc = exec_evaluation(sess, X, Y, epoch_num, data, batch_size, accuracy)

    sess.close()

    return acc


if __name__ == "__main__":
    start = time.time()
    acc = main(parse_args())
    print('精度 : ' + str(acc))
    print('時間 : ' + str(time.time() - start) + 's')
