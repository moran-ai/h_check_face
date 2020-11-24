import tensorflow as tf
import cv2 as cv
import os
import numpy as np
from tensorflow.keras import layers

def check_f(path_):
    if not os.path.join(path_):
        os.mkdir(path_)

    # 使用opencv调用摄像头拍摄图片,并保存
    # 打开摄像头
    cap = cv.VideoCapture(0)

    width, height, w = 64, 64, 360
    print('width：', width)
    print('height：', height)
    count = 1
    # 设定窗口的大小
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)

    crop_w_start = (width-w)//2
    crop_h_start = (height-w)//2

    # 打开摄像头
    while True:
        ret, frame = cap.read()
        # 相框大小
        frame = frame[crop_h_start: crop_h_start + w, crop_w_start: crop_w_start + w]  # 展示相框

        frame = cv.flip(frame, 1, dst=None)
        # 展示摄像头中的内容
        cv.imshow("capture", frame)
        action = cv.waitKey(1) & 0xFF
        if action == ord('q'):
            break
        if action == ord('p'):
            cv.imwrite("%s/%d.jpg" % (path_, count), cv.resize(frame, (64, 64), interpolation=cv.INTER_AREA))
            print(f'{path_}：第{count}图片')
            count += 1
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    check_f(path_='./h')

def img_picr(saveDir='../hu_bo_'):
  '''
  调用电脑摄像头来拍摄图片，并保存在本地
  '''
  # 如果目录不存在，就进行目录的创建
  if not os.path.exists(saveDir):
    os.makedirs(saveDir)

  # 设置图片计数开始
  count=1
  # 开启摄像头
  cap=cv.VideoCapture(0)
  # 设置窗口的宽，高
  width,height,w= 64,64,360
  # 设置窗口大小
  cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

  # 设置相框大小
  crop_w_start=(width-w)//2
  crop_h_start=(height-w)//2

  print('width: ',width)
  print('height: ',height)
  while True:
    ret,frame = cap.read() #获取相框
    # 相框大小
    frame = frame[crop_h_start: crop_h_start + w, crop_w_start: crop_w_start + w] #展示相框
    # 前置摄像头获取的画面是非镜面的，即左手会出现在画面的右侧，此处使用flip进行水平镜像处理
    frame=cv.flip(frame, 1, dst=None)
    # 展示摄像头中的内容
    cv.imshow("capture", frame)
    action=cv.waitKey(1) & 0xFF

    # 进行拍照
    if action == ord('p'):
      cv.imwrite("%s/%d.jpg" % (saveDir,count), cv.resize(frame, (64, 64),interpolation=cv.INTER_AREA))
      print(u"%s: %d 张图片" % (saveDir,count))
      count+=1
    # 退出
    if action == ord('q'):
      break
  cap.release() #释放摄像头
  cv.destroyAllWindows() #丢弃窗口

# 主程序
# if __name__=='__main__':
#   img_picr(saveDir='../h')
