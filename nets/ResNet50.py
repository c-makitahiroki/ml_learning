import tensorflow as tf

def weight_variable(shape, name=None):
    # 標準偏差が0.1の切断正規分布からshapeで指定された形のテンソルを生成し、initialへ代入
    initial = tf.truncated_normal(shape, stddev=0.1)

    # 初期のテンソルがinitialの変数tf.Variableを返す
    return tf.Variable(initial, name=name)


def softmax_layer(input, shape):
    # shapeで指定される形の重みをfc_wへ代入
    fc_w = weight_variable(shape)

    # shape[1]で指定される形の重みをfc_bに代入
    # 初期値はゼロベクトル
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    # 全結合後、ソフトマックスを計算
    fc_h = tf.nn.softmax(tf.matmul(input, fc_w) + fc_b)

    return fc_h


def conv_layer(input, filter_shape, stride):
    # 入力データのチャンネル数をinput_channelsに代入
    input_channels = input.get_shape().as_list()[3]

    # Batch Normalization
    # チャンネルごとに平均meanと分散varを計算
    mean, var = tf.nn.moments(input, axes=[0, 1, 2])

    # Batch Normalizationに使用する学習パラメータbetaとgammaを準備
    beta = tf.Variable(tf.zeros([input_channels]), name="beta")
    gamma = weight_variable([input_channels], name="gamma")

    # Batch Normalization実施
    batch_norm = tf.nn.batch_norm_with_global_normalization(input, mean, var, beta, gamma, 0.001,
                                                            scale_after_normalization=True)

    # 活性化関数としてReLU関数を使用
    out_relu = tf.nn.relu(batch_norm)

    # 畳み込み層
    filter_ = weight_variable(filter_shape)

    # 畳み込み層の出力をoutに代入
    out = tf.nn.conv2d(out_relu, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")

    return out


def residual_block(input, output_depth, stride=1, projection=False):
    # 入力データのチャンネル数をinput_depthに代入
    input_depth = input.get_shape().as_list()[3]

    # BatchNormalization + Relu + 畳み込みを3セット
    conv1 = conv_layer(input, [1, 1, input_depth, int(output_depth / 4)], stride)
    conv2 = conv_layer(conv1, [3, 3, int(output_depth / 4), int(output_depth / 4)], stride)
    conv3 = conv_layer(conv2, [1, 1, int(output_depth / 4), output_depth], stride)

    # 入力と出力が違う場合はチャンネル数をそろえる
    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(input, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = input

    res = conv3 + input_layer

    return res


def resnet50(input):
    layers = []

    # Residual Blockに入る前に、１つ畳み込み層とmax poolingを通す
    with tf.variable_scope('conv1'):
        conv1 = conv_layer(input, [7, 7, 1, 16], 1)
        max_pooling = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 1, 1, 1], padding="SAME")
        layers.append(conv1)
        layers.append(max_pooling)

    # residual blockの総数は3個
    # 出力のshapeは[batch_size, 28, 28, 16]
    with tf.variable_scope('conv2'):
        conv2_1 = residual_block(layers[-1], 16)
        conv2_2 = residual_block(conv2_1, 16)
        conv2_3 = residual_block(conv2_2, 16)
        layers.append(conv2_1)
        layers.append(conv2_2)
        layers.append(conv2_3)

    assert conv2_3.get_shape().as_list()[1:] == [28, 28, 16]

    # residual blockの総数は4個
    # 出力のshapeは[batch_size, 28, 28, 32]
    with tf.variable_scope('conv3'):
        conv3_1 = residual_block(layers[-1], 32, stride=1)
        conv3_2 = residual_block(conv3_1, 32)
        conv3_3 = residual_block(conv3_2, 32)
        conv3_4 = residual_block(conv3_3, 32)
        layers.append(conv3_1)
        layers.append(conv3_2)
        layers.append(conv3_3)
        layers.append(conv3_4)

    assert conv3_4.get_shape().as_list()[1:] == [28, 28, 32]

    # residual blockの総数は6個
    # 出力のshapeは[batch_size, 28, 28, 64]
    with tf.variable_scope('conv4'):
        conv4_1 = residual_block(layers[-1], 64, stride=1)
        conv4_2 = residual_block(conv4_1, 64)
        conv4_3 = residual_block(conv4_2, 64)
        conv4_4 = residual_block(conv4_3, 64)
        conv4_5 = residual_block(conv4_4, 64)
        conv4_6 = residual_block(conv4_5, 64)
        layers.append(conv4_1)
        layers.append(conv4_2)
        layers.append(conv4_3)
        layers.append(conv4_4)
        layers.append(conv4_5)
        layers.append(conv4_6)

    assert conv4_6.get_shape().as_list()[1:] == [28, 28, 64]

    # residual blockの総数は3個
    # 出力のshapeは[batch_size, 28, 28, 128]
    with tf.variable_scope('conv5'):
        conv5_1 = residual_block(layers[-1], 128)
        conv5_2 = residual_block(conv5_1, 128)
        conv5_3 = residual_block(conv5_2, 128)
        layers.append(conv5_1)
        layers.append(conv5_2)
        layers.append(conv5_3)

    assert conv5_3.get_shape().as_list()[1:] == [28, 28, 128]

    with tf.variable_scope('fc'):
        # batch_sizeとチャンネル数ごとに平均をとる
        global_pool = tf.reduce_mean(layers[-1], [1, 2])

        assert global_pool.get_shape().as_list()[1:] == [128]

        # 全結合 + ソフトマックス
        out = softmax_layer(global_pool, [128, 10])
        layers.append(out)

    return layers[-1]
