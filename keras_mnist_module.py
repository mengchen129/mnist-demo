from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import load_model
from PIL import Image
import numpy as np
import os

network = None
n_type = None


# 训练网络
# dense - 密集连接网络
# conv - 卷积神经网络
def init_network(network_type='dense'):
    global network, n_type

    n_type = network_type

    # 尝试从已保存的训练网络中加载
    saved_file = f'models/{network_type}.h5'
    if os.path.isfile(saved_file):
        network = load_model(saved_file)
        return

    # 网络下载数据文件要访问 googleapi，此处使用已经离线下载好的数据文件
    local_path = '/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/keras/datasets/mnist.npz'
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=local_path)
    network = models.Sequential()

    if network_type == 'dense':
        train_images = train_images.reshape((60000, 28 * 28))
        train_images = train_images.astype('float32') / 255

        network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        network.add(layers.Dense(10, activation='softmax'))
    elif network_type == 'conv':
        train_images = train_images.reshape((60000, 28, 28, 1))
        train_images = train_images.astype('float32') / 255

        network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        network.add(layers.MaxPooling2D((2, 2)))
        network.add(layers.Conv2D(64, (3, 3), activation='relu'))
        network.add(layers.MaxPooling2D((2, 2)))
        network.add(layers.Conv2D(64, (3, 3), activation='relu'))
        network.add(layers.Flatten())
        network.add(layers.Dense(64, activation='relu'))
        network.add(layers.Dense(10, activation='softmax'))

    train_labels = to_categorical(train_labels)
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    # 保存模型
    network.save(f'models/{network_type}.h5')


# predict 指定某个图像文件，返回完整的识别结果
def predict(img_file):
    global network, n_type

    img = Image.open(img_file)

    # 转成灰度图像，具体算法可见注释
    img = img.convert('L')

    # 缩放到 28*28
    img = img.resize((28, 28))

    np_img = np.array(img)

    # 将黑白互换（反色），因为前端传入的是白底黑字，而训练的图是黑底白字
    np_img = 255 - np_img

    # 增加一层维度，以适配网络预测输入格式
    np_img_arr = np.array([np_img])

    if n_type == 'dense':
        # 将二维转成一维（28*28 => 1*784）
        np_img_arr = np_img_arr.reshape((1, 28 * 28))
    elif n_type == 'conv':
        # 将二维转成三维（28*28 => 28*28*1）
        np_img_arr = np_img_arr.reshape((1, 28, 28, 1))

    # 处理成值在 0-1 内（之前是 0-255）
    np_img_arr = np_img_arr.astype('float32') / 255

    predict_result = network.predict(np_img_arr)
    print(predict_result)
    return predict_result[0].tolist()

