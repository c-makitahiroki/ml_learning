import tensorflow as tf
from utils.download_image import read_mnist_datasests


def exec_train(train_op, X, Y, epoch_num, data, batch_size, loss, lr, learning_rate):
    train_step_num = int(data.train.num_examples / batch_size)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    # 学習
    for j in range(epoch_num):
        for i in range(train_step_num):
            x_train, y_train = read_mnist_datasests(data, batch_size, "train")
            if i == 0:
                saver.save(sess, "./models/model.ckpt%d" % j)  # epoch毎にモデルを作成
                loss_val = sess.run(loss, feed_dict={X: x_train, Y: y_train})
                print(loss_val)
            feed_dict = {X: x_train, Y: y_train, lr: learning_rate}
            sess.run([train_op], feed_dict=feed_dict)

    return sess
