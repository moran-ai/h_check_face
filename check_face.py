import cv2
from predict import infer

def generator():
    path = "./image"
    # 创建一个级联分类器
    face_casecade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('人脸识别')
    while (True):
        # 读取一帧图像
        ret, frame = camera.read()
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测q
            face = face_casecade.detectMultiScale(gray_img, 1.3, 5)

            for (x, y, w, h) in face:
                # 在原图上绘制矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 调整图像大小
                new_frame = cv2.resize(gray_img[y:y + h, x:x + w], (64, 64))

                # 识别l人脸
                cv2.imwrite("test.png", new_frame)
                res = infer("test.png")
                cv2.putText(frame, res, (180, 320), cv2.FONT_HERSHEY_COMPLEX, 1, (180, 100, 255), 2, cv2.LINE_AA)

            cv2.imshow('Dynamic', frame)
            # 按下q键退出
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()
generator()
