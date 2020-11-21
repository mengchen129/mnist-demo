from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from PIL import Image
import numpy as np

network = None
test_images = []
test_labels = []


def init_network():
    global network, test_images, test_labels

    # 网络下载数据文件要访问 googleapi，此处使用已经离线下载好的数据文件
    local_path = '/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/keras/datasets/mnist.npz'
    (train_images, train_labels), (orig_test_images, test_labels) = mnist.load_data(path=local_path)

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = orig_test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=5, batch_size=128)


# predict 指定某个图像文件，返回完整的识别结果
def predict(img_file):
    global network

    img = Image.open(img_file)
    img = img.convert('L')
    img = img.resize((28, 28))
    np_img = np.array(img)
    np_img = 255 - np_img
    np_img_arr = np.array([np_img])
    np_img_arr = np_img_arr.reshape((1, 28 * 28))
    np_img_arr = np_img_arr.astype('float32') / 255

    predict_result = network.predict(np_img_arr)
    print(predict_result)
    return predict_result[0].tolist()

    # 找出概率最大值的下标
    # max_index = np.argmax(np.array(predict_result[0]))
    # print(max_index)
    # return max_index


if __name__ == '__main__':
    init_network()
    result = predict(2048)
    maxIndex = np.argmax(np.array(result[0]))
    print(maxIndex)