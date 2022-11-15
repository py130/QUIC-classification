# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import (BatchNormalization, Convolution1D, MaxPooling1D, Reshape, Flatten)
import numpy as np
import time

# 标签种类，在第一次实验中，为5种应用: google doc/ google drive / google music / google search / youtube
LABELNUM = 5


class CNN1D:

    def __init__(self, enabled):
        self.name = 'CNN1D'
        self.enabled = enabled
        self.model = self.build()

    def build(self):
        model = tf.keras.Sequential()
        model.add(Reshape((100, 1), input_shape=(1, 100)))
        model.add(Convolution1D(16, 3, strides=3, padding='same', input_shape=(100, 1), activation='tanh'))  # 第一个卷积层
        model.add(Dropout(0.5))  # 将经过第一个卷积层后的输出数据按照0.5的概率随机置零，也可以说是灭活
        # 添加批量标准层，将经过dropout的数据形成正态分布，有利于特征集中凸显
        model.add(BatchNormalization())
        # 添加池化层，池化核大小为2步长为2，padding数据尾部补零。池化层不需要设置通道数，但卷积层需要
        model.add(MaxPooling1D(2, strides=2, padding='same'))
        model.add(Convolution1D(32, 3, padding='same', activation='tanh'))  # 第二个卷积层
        model.add(BatchNormalization())
        model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第三个卷积层
        model.add(BatchNormalization())
        model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第四个卷积层
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2, strides=2, padding='same'))
        model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第五个卷积层
        model.add(BatchNormalization())
        model.add(Convolution1D(64, 3, padding='same', activation='tanh'))  # 第六个卷积层
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2, strides=2, padding='same'))
        model.add(Flatten())  # 将经过卷积和池化的数据展平，具体操作方式可以理解为，有n个通道的卷积输出，将每个通道压缩成一个数据，这样展平后就会出现n个数据
        model.add(Dense(100))
        model.add(Dense(LABELNUM, activation='softmax', name='b'))  # 最后一层的参数设置要和标签种类一致，而且激活函数采取分类器softmax
        # print(model.summary())  # 模型小结，在训练时可以看到网络的结构参数
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.build(input_shape=(None, 1, 100))
        return model

    def predict(self, inputs):
        output = self.model.predict(inputs)
        res = [np.argmax(output[i]) for i in range(len(output))]
        return res

    def change_state(self):
        self.enabled = not self.enabled

    def is_enabeld(self):
        return self.enabled

def change_type(data):
    xx = []
    for i in range(len(xdata)):
        x = []

        for k in range(len(xdata[i])):
            x.extend(xdata[i][k])
        xx.append(x)
    return xx


# -------------configure----------------
BATCH_SIZE = 20
# 默认 xdataPath="datanpy/xdata.npy"，读取示例数据，只有 150 个会话的数据
# 可以选择用 xdataPath="datanpy-all/xdata.npy", ydata="datanpy-all/ydata.npy" ，读取到完整的训练数据，包含了 6000+ 会话数据，准确率更高但是训练更慢
xdataPath = "datanpy/xdata.npy"
ydataPath = "datanpy/ydata.npy"
# -------------configure----------------


# 读取处理好的数据

# 如果用后者，建议调大 BATCH_SIZE 变量，例如调为100
xdata = np.load(xdataPath)
# 标签：shape为(会话数)，每个值为一个标签，从0-4不等，代表5种协议，详见processData.py的labelDict字典
ydata = np.load(ydataPath)
# 将数据转为numpy数组
train_x = np.array(xdata)
# 对标签进行one-hot编码
train_y = to_categorical(ydata)


# 将所有数据按照0.4：0.6的比例划分为验证集、训练集
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.4, random_state=2018, stratify=train_y)

# 模型初始化
model = CNN1D(True)
# 开始时间
start = time.time()
# 模型训练
history = model.model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), batch_size=BATCH_SIZE)
# 保存模型
stored_model_path = "stored_model/weights_cnn1d_simple.h5"
if xdataPath == "datanpy/xdata.npy":
    stored_model_path = "stored_model/weights_cnn1d_simple.h5"
elif xdataPath == "datanpy-all/xdata.npy":
    stored_model_path = "stored_model/weights_cnn1d.h5"
model.model.save_weights(stored_model_path)

# 结束时间
end = time.time()
# 计算模型训练总时间
print("用时：", end - start)

# 绘制图像
import matplotlib.pyplot as plt  # 导入plt画图工具

plt.figure(figsize=(8, 6))
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 画出训练的损失函数曲线
plt.plot(history.history['loss'], label="Train_loss")
# 画出验证的损失函数曲线
plt.plot(history.history['val_loss'], label='Val_loss')
plt.xlabel('Epochs ', fontsize=20)  # x轴名称
plt.ylabel('Accuracy and Loss', fontsize=20)  # y轴名称
plt.legend()
plt.show()

# 画出训练的准确率曲线
plt.plot(history.history['accuracy'], label='Train_acc', )
# 画出验证的准确率曲线
plt.plot(history.history['val_accuracy'], label='Val_acc')
plt.legend()
plt.xlabel('Epochs ', fontsize=20)  # x轴名称
plt.ylabel('Accuracy and Loss', fontsize=20)  # y轴名称
plt.show()
#
