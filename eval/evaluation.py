from utils.download_image import read_mnist_datasests

import tensorflow as tf


def exec_evaluation(sess, X, Y, epoch_num, data, batch_size, accuracy):
    test_step_num = int(data.test.num_examples / batch_size)
    accs = []
    for j in range(epoch_num):
        for i in range(test_step_num):
            x_test, y_test = read_mnist_datasests(data, batch_size, "test")
            acc = sess.run([accuracy], feed_dict={X: x_test, Y: y_test})
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)
            accs.append(acc[0])

    return sum(accs) / len(accs)
