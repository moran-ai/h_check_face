import os

import tensorflow as tf
import random
import pathlib

from tensorflow.keras import layers, models

AUTOTUNE = tf.data.experimental.AUTOTUNE
# Imagenet标准化
MEAN_RGB = [0.485, 0.456, 0.406]
STED_RGB = [0.229, 0.224, 0.225]
img_height = 64
img_width = 64

## 定义ImageNet标准化方法
def _normalize(image, mean, std):
    offset = tf.constant(mean, shape=[1, 1, 3])
    image -= offset

    scale = tf.constant(std, shape=[1, 1, 3])
    image /= scale
    return image

## 定义图像从路径到tensor变量的读取方法
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image)
    # image = tf.reshape(image, shape=(img_height, img_width, 3))
    image = tf.cast(image, tf.float32)  / 255.0
    return image, label

# 定义数据增强方法
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.flip_up_down(image)
    label = tf.one_hot(label, 12)
    return image, label

# 数据加载方法
def get_paths_labels(path, class_names):
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = class_names
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    return all_image_paths, all_image_labels

def main():
    class_names = os.listdir('../test/small_img_gray')
    BATCH_SIZE = 32

    # tensorflow.data.dataset方法将数据空间中的数据集加载进来，同时训练和测试集分别对应不同的预处理方法
    # 加载训练集将预处理方法作为参数输入
    TRAIN_DATA_DIR = '../test/small_img_gray'
    TEST_DATA_DIR = '../test/small_img_gray'

    ## 给定函数获取训练与测试图像
    tr_image_paths, tr_image_labels = get_paths_labels(TRAIN_DATA_DIR, class_names)
    ts_image_paths, ts_image_labels = get_paths_labels(TEST_DATA_DIR, class_names)

    ##通过 tensorflow Dataset 中的from_tensor_slices方法构建训练tr_dataset与ts_dataset
    tr_dataset = tf.data.Dataset.from_tensor_slices((tr_image_paths, tr_image_labels))
    ts_dataset = tf.data.Dataset.from_tensor_slices((ts_image_paths, ts_image_labels))

    tr_dataset = tr_dataset.map(load_and_preprocess_image)
    tr_dataset = tr_dataset.map(augment)
    # ## 对测试集进行标准化处理
    ts_dataset = ts_dataset.map(load_and_preprocess_image)
    ts_dataset = ts_dataset.map(augment)

    ## 使用Sequential构造模型
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(filters=96, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(len(class_names), activation="softmax"))
    model.summary()

    # 模型编译
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 调用fit函数进行训练
    history = model.fit(tr_dataset.batch(BATCH_SIZE),
                        epochs=20,
                        verbose=1)

    ## 计算测试集准确率
    acc = None
    acc = model.evaluate(ts_dataset.batch(BATCH_SIZE))
    print("The accuracy {}".format(acc[1]))

    MODEL_DIR = '../test/'
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    model.save(export_path + ".hdf5")

if __name__ == '__main__':
    main()


# Epoch 1/20
# 31/31 [==============================] - 4s 126ms/step - loss: 1.6910 - accuracy: 0.4255
# Epoch 2/20
# 31/31 [==============================] - 4s 128ms/step - loss: 0.3556 - accuracy: 0.9099
# Epoch 3/20
# 31/31 [==============================] - 4s 124ms/step - loss: 0.0812 - accuracy: 0.9679
# Epoch 4/20
# 31/31 [==============================] - 4s 125ms/step - loss: 0.0355 - accuracy: 0.9917
# Epoch 5/20
# 31/31 [==============================] - 4s 128ms/step - loss: 0.0206 - accuracy: 0.9948
# Epoch 6/20
# 31/31 [==============================] - 4s 127ms/step - loss: 0.0264 - accuracy: 0.9917
# Epoch 7/20
# 31/31 [==============================] - 4s 127ms/step - loss: 0.0038 - accuracy: 0.9990
# Epoch 8/20
# 31/31 [==============================] - 4s 127ms/step - loss: 0.0046 - accuracy: 0.9979
# Epoch 9/20
# 31/31 [==============================] - 4s 127ms/step - loss: 0.0179 - accuracy: 0.9928
# Epoch 10/20
# 31/31 [==============================] - 4s 128ms/step - loss: 0.0273 - accuracy: 0.9907
# Epoch 11/20
# 31/31 [==============================] - 4s 127ms/step - loss: 0.0358 - accuracy: 0.9876
# Epoch 12/20
# 31/31 [==============================] - 4s 129ms/step - loss: 0.0436 - accuracy: 0.9855
# Epoch 13/20
# 31/31 [==============================] - 4s 128ms/step - loss: 0.0098 - accuracy: 0.9959
# Epoch 14/20
# 31/31 [==============================] - 4s 129ms/step - loss: 9.1002e-04 - accuracy: 1.0000
# Epoch 15/20
# 31/31 [==============================] - 4s 128ms/step - loss: 2.9859e-04 - accuracy: 1.0000
# Epoch 16/20
# 31/31 [==============================] - 4s 128ms/step - loss: 1.0823e-04 - accuracy: 1.0000
# Epoch 17/20
# 31/31 [==============================] - 4s 130ms/step - loss: 6.3987e-05 - accuracy: 1.0000
# Epoch 18/20
# 31/31 [==============================] - 4s 130ms/step - loss: 6.8762e-05 - accuracy: 1.0000
# Epoch 19/20
# 31/31 [==============================] - 4s 130ms/step - loss: 4.8130e-05 - accuracy: 1.0000
# Epoch 20/20
# 31/31 [==============================] - 4s 131ms/step - loss: 3.9917e-05 - accuracy: 1.0000
# 31/31 [==============================] - 1s 42ms/step - loss: 3.7989e-05 - accuracy: 1.0000
# The accuracy 1.0
