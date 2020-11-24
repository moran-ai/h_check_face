# import cv2
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
#
# def LoadImages(data):
#     '''
#     加载图片数据用于训练
#     params:
#         data:训练数据所在的目录，要求图片尺寸一样
#     ret:
#         images:[m,height,width]  m为样本数，height为高，width为宽
#         names：名字的集合
#         labels：标签
#     '''
#     images = []
#     names = []
#     labels = []
#     label = 0
#
#     # 遍历所有文件夹
#     for subdir in os.listdir(data):
#         subpath = os.path.join(data, subdir)
#         # print('path',subpath)
#         # 判断文件夹是否存在
#         if os.path.isdir(subpath):
#             # 在每一个文件夹中存放着一个人的许多照片
#             names.append(subdir)
#             # 遍历文件夹中的图片文件
#             for filename in os.listdir(subpath):
#                 imgpath = os.path.join(subpath, filename)
#                 img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
#                 # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 # cv2.imshow('1',img)
#                 # cv2.waitKey(0)
#                 images.append(img)
#                 labels.append(label)
#             label += 1
#     images = np.asarray(images)
#     # names=np.asarray(names)
#     labels = np.asarray(labels)
#     return images, labels, names
#
# # 检验训练结果
# def FaceRec(data):
#     # 加载训练的数据
#     X, y, names = LoadImages(data)
#     print(X.shape)
#     print(y.shape)
#     # model = load_model('1.hdf5')
#     # model.fit(X, y)
#
#     # 打开摄像头
#     camera = cv2.VideoCapture(0)
#     cv2.namedWindow('Dynamic')
#
#     # 创建级联分类器
#     # face_casecade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
#
#     while (True):
#         # 读取一帧图像
#         # ret:图像是否读取成功
#         # frame：该帧图像
#         ret, frame = camera.read()
#         # 判断图像是否读取成功
#         # print('ret',ret)
#         if ret:
#             # 转换为灰度图
#             gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#             # 利用级联分类器鉴别人脸
#             faces = face_casecade.detectMultiScale(gray_img, 1.3, 5)
#
#             # 遍历每一帧图像，画出矩形
#             for (x, y, w, h) in faces:
#                 frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 蓝色
#                 roi_gray = gray_img[y:y + h, x:x + w]
#
#                 try:
#                     # 将图像转换为宽92 高112的图像
#                     # resize（原图像，目标大小，（插值方法）interpolation=，）
#                     roi_gray = cv2.resize(roi_gray, (92, 112), interpolation=cv2.INTER_LINEAR)
#                     params = model.predict(roi_gray)
#                     print('Label:%s,confidence:%.2f' % (params[0], params[1]))
#                     '''
#                     putText:给照片添加文字
#                     putText(输入图像，'所需添加的文字'，左上角的坐标，字体，字体大小，颜色，字体粗细)
#                     '''
#                     cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
#                 except:
#                     continue
#
#             cv2.imshow('Dynamic', frame)
#
#             # 按下q键退出
#             if cv2.waitKey(100) & 0xff == ord('q'):
#                 break
#     camera.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     data = '../test/small_img_gray'
#     FaceRec(data)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# 图像ImageNet标准化
MEAN_RGB=[0.485, 0.456, 0.406]
STED_RGB=[0.229, 0.224, 0.225]

# 指定类别名称
label_names = ['caoqilin', 'fanzichao', 'hebo', 'hexianbin', 'hu_bo',
               'huhuijuan', 'jiangpenghui', 'liuzhenyu', 'panhonghai',
               'wuguipeng', 'zhangjinwei', 'zhangmin']
print(label_names)

img_height = 64
img_width = 64

# 图像ImageNet标准化
def _normalize(image):
    offset = tf.constant(MEAN_RGB, shape = [1,1,3])
    image -= offset

    scale = tf.constant(STED_RGB, shape=[1,1,3])
    image /= scale
    return image

# 利用tensorflow 接口通过路径获取图像
def load_image(image_path):
    """ 调用Tensorflow api加载图像，并统一图像尺寸 """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)
    image = tf.reshape(image, shape=(img_height, img_width, 1))
    image = tf.cast(image, tf.float32)
    return image

# 利用tensorflow将图像转化为数组，并满足模型预测要求
def image2image_array(image):
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = _normalize(image_array)
    #利用tensorflow接口，创建batch
    image_array = tf.reshape(image_array, shape=(-1, img_height, img_width, 1))
    return image_array

# 加载预训练模型结构与权重
## 加载model文件夹中模型
model = load_model('../test/1.hdf5')
model.summary()

## 构建推理函数并完成指定图像的识别
def infer(image_path, model=model, label_names=label_names):
    img = load_image(image_path)
    img_array = image2image_array(img)

    # 计算得到模型输出结果
    la = model.predict(img_array)
    result = np.argmax(la)
    if result >= 0 and result <= 12:
        print("分类结果为： {}".format(label_names[result - 1]))
        return label_names[result - 1]
    else:
        return 'not found face, please check your face'

if __name__ == '__main__':
    for paths in os.listdir('test/small_img_gray/zhangjinwei/'):
        path = os.path.join('test/small_img_gray/zhangjinwei/', paths)
        infer(path)
