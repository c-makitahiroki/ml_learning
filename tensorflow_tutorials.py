from utils.download_image import download_mnist_datasets

import tensorflow as tf


def define_variables():
    x = tf.placeholder(tf.float32, [None, 784])  # MNISTの画像（784次元）が入る箱を定義。[任意,784]の2Dテンソル
    W = tf.Variable(tf.zeros([784, 10]))  # 重み。ゼロで初期化
    b = tf.Variable(tf.zeros([10]))  # バイアス。ゼロで初期化

    return x, W, b


def predict(x, y, y_, mnist, sess):
    correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def model_deploy(x, W, b):
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 学習率0.5で勾配降下法で学習する。

    return y, y_, train_step


def main():
    # データのロード
    mnist = download_mnist_datasets("fashion_mnist")

    # 変数の定義
    x, W, b = define_variables()

    # モデルの定義
    y, y_, train_step = model_deploy(x, W, b)

    # セッションの定義
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 学習
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 評価
    predict(x, y, y_, mnist, sess)


if __name__ == "__main__":
    main()
