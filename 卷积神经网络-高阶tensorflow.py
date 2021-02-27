import gzip
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


def read_mnist(images_path, labels_path):
    with gzip.open("MNIST_data/" + labels_path, 'rb') as labelsFile:
        y=np.frombuffer(labelsFile.read(),dtype=np.uint8,offset=8)

    with gzip.open("MNIST_data/" +images_path, 'rb') as imagesFile:
        X=np.frombuffer(imagesFile.read(),dtype=np.uint8,offset=16).reshape(len(y),784).reshape(len(y),28,28,1)

    return X,y

train={}
test={}

# 获取训练集
train['X'], train['y']=read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
# 获取测试集
test['X'], test['y'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

print(train['X'].shape, train['y'].shape, test['X'].shape, test['y'].shape)

plt.imshow(train['X'][0].reshape(28,28), cmap=plt.cm.gray_r)

# 样本 padding 填充
X_train = np.pad(train['X'], ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(test['X'], ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# 标签独热编码
y_train = np.eye(10)[train['y'].reshape(-1)]
y_test = np.eye(10)[test['y'].reshape(-1)]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

plt.imshow(X_train[0].reshape(32,32),cmap=plt.cm.gray_r)


model=tf.keras.Sequential()


# 卷积层 6个5*5的卷积核，步长为1，relu为激活函数，第一层需指定input_shape
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1),
                                 activation='relu', input_shape=(32, 32, 1)))

# 池化层 平均池化，池化窗口默认为2
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=2))

# 卷积层 16个5*5卷积核，步为1，relu激活
model.add(tf.keras.layers.Conv2D(filters=16 ,kernel_size=(5,5),strides=(1, 1),
                                 activation='relu'))

# 池化层 平均池化层，池化窗口为2
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),strides=2))

# 需展平后才能与全连接层相连----Flatten展平
model.add(tf.keras.layers.Flatten())

# 全连接层，输出为120，relu为激活函数
model.add(tf.keras.layers.Dense(units=120,activation='relu'))

# 全连接层，输出为84，relu为激活函数
model.add(tf.keras.layers.Dense(units=84, activation='relu'))

# 全连接层，输出为10，softmax为激活函数
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# 查看网络结构
model.summary()

# 编译模型，Adam优化器，多分类交叉熵损失函数，准确度评估
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型训练及评估
model.fit(X_train, y_train, batch_size=64, epochs=2, validation_data=(X_test, y_test))
