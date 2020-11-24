import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print('tf版本：', tf.__version__)

# 将图片变为灰度图
# for i in os.listdir('../hu_bo_'):
#     data = cv2.imread('../hu_bo_/' + i, 0)
#     cv2.imwrite('hu_bo/' + i, data)

img_width = 64
img_height = 64

class_names = [
    'fanzichao', 'zhangjingwei', 'chentao', 'wuguipeng', 'hubo', 'caoqilin', 'hexianbin'
]
print(len(class_names))

# 加载数据集
for i in os.listdir('train'):
    image_ = tf.io.read_file('train/' + i)
    image1 = tf.io.decode_image(image_)
    print(image1)

# # 数据集的路径
# img_path = tf.data.Dataset.list_files('train')
# for i in img_path:
#     print(i)
#
# # 加载数据集
# def load_image(image_file, is_train):
#     image = tf.io.read_file(image_file)
#     image = tf.io.decode_image(image)
#     return image
#
# # 迭代器for循环
# train_iter = iter(img_path)
# train_data = []
# for x in train_iter:
#     train_data.append(load_image(x, True))
# train_data = tf.stack(train_data, axis=0)
# print('train:', train_data.shape)
#
# img_path = tf.data.Dataset.from_tensor_slices(train_data)
# img_path = img_path.shuffle(60).batch(2)

model = Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train=train_data, y_train=img_path, epochs=5, verbose=2)

acc, loss = model.evaluate(x_test, y_test)
print(f'The accuracy：{acc}')
print(f'The loss：{loss}')
