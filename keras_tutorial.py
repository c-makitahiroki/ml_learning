import keras

from utils.download_image import download_mnist_for_keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer


def processing_mnist(x_train, y_train, x_test, y_test):
    # MNISTデータの加工
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def model_deploy():
    # 学習モデルの生成
    model = Sequential()
    model.add(InputLayer(input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train(model, x_train, y_train, x_test, y_test, epochs=20, batch_size=128):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test))

    return history


def evaluation(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)

    return score


def main():
    # データのダウンロード
    x_train, y_train, x_test, y_test = download_mnist_for_keras()
    x_train, y_train, x_test, y_test = processing_mnist(x_train, y_train, x_test, y_test)

    # モデルの生成
    model = model_deploy()

    # 学習
    history = train(model, x_train, y_train, x_test, y_test)

    # 評価
    score = evaluation(model, x_test, y_test)

    # 結果の表示
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


if __name__ == "__main__":
    main()
