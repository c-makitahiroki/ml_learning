from utils.download_image import download_mnist_datasets, read_mnist_datasests
from nets.ResNet50 import resnet50

import tensorflow as tf
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=str, default=128)
    parser.add_argument("--epoch_num", type=str, default=10)
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--datatype", type=str, default="fashion_mnist")

    args = parser.parse_args()

    return args


def main(args):
    batch_size = args.batch_size
    epoch_num = args.epoch_num

    X = tf.placeholder("float", [batch_size, 28, 28, 1])
    Y = tf.placeholder("float", [batch_size, 10])
    learning_rate = tf.placeholder("float", [])

    # resnet
    net = resnet50(X)

    # 損失関数と学習メソッドの定義
    cross_entropy = -tf.reduce_sum(Y * tf.log(net))
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = opt.minimize(cross_entropy)




    # セッションの初期化
    sess = tf.Session()

    # Tensorboard
    with tf.name_scope('summary'):
        tf.summary.scalar('loss', cross_entropy)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.initialize_all_variables())

    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    data = download_mnist_datasets(args.datatype)

    train_step_num = int(data.train.num_examples / batch_size)
    test_step_num = int(data.test.num_examples / batch_size)


    # 学習
    for j in range(epoch_num):
        for i in range(train_step_num):
            x_train, y_train = read_mnist_datasests(data, batch_size, "train")
            feed_dict = {X: x_train, Y: y_train, learning_rate: args.learning_rate}
            sess.run([train_op], feed_dict=feed_dict)

    # テスト
    accs = []
    for j in range(epoch_num):
        for i in range(test_step_num):
            x_test, y_test = read_mnist_datasests(data, batch_size, "test")
            acc = sess.run([accuracy], feed_dict={X: x_test, Y: y_test})
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)
            accs.append(acc[0])
    sess.close()

    return sum(accs) / len(accs)


if __name__ == "__main__":
    start = time.time()
    acc = main(parse_args())
    print('精度 : ' + str(acc))
    print('時間 : ' + str(time.time() - start) + 's')
