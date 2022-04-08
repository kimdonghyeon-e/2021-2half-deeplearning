# 필요 라이브러리 선언
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from PIL import ImageOps, Image
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tkinter import *
import tkinter.ttk
import PIL.Image
import PIL.ImageTk
from PIL import ImageTk,Image
import threading 
import time
import Jetson.GPIO as GPIO

# 텐서플로 메모리부족문제 해결위한 코드
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# LED 핀 지정
redPin = 11
yellowPin = 12
greenPin = 13

# GPIO출력설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(redPin, GPIO.OUT)
GPIO.setup(yellowPin, GPIO.OUT)
GPIO.setup(greenPin, GPIO.OUT)

# 학습모델 임포트
opt = optimizers.Adam(learning_rate=0.0001)

with open('covid_classifier_model.json', 'r') as json_file:
    json_savedModel= json_file.read()

model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('covid_classifier_weights.h5')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

# 이미지 사이즈 조절용 함수
def read_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
    img = cv2.resize(img, (256, 256)) 
    img = np.array(img, dtype='float32') 
    img = img/255 
    img = np.expand_dims(img, axis=0)
    return img

# 결과값 확인 함수
def viewopen(): 
        view = Tk()
        view.title('covid check_view image')
        view.geometry('700x700')
        chkvar = IntVar()
        radio1 = tkinter.Radiobutton(view, text="코로나 확진", variable=chkvar, value=0)
        radio1.pack()

        chkvar2 = IntVar()
        radio2 = tkinter.Radiobutton(view, text="일반 폐렴", variable=chkvar2, value=1)
        radio2.pack()

        chkvar3 = IntVar()
        radio3 = tkinter.Radiobutton(view, text="정상", variable=chkvar3, value=2)
        radio3.pack()
        
        if(num==0):
            radio1.select()
            GPIO.output(redPin, GPIO.HIGH)
            GPIO.output(yellowPin, GPIO.LOW)
            GPIO.output(greenPin, GPIO.LOW)
            print("코로나19")
        elif(num==1):
            radio2.select()
            GPIO.output(redPin, GPIO.LOW)
            GPIO.output(yellowPin, GPIO.HIGH)
            GPIO.output(greenPin, GPIO.LOW)
            print("일반폐렴")
        else:
            GPIO.output(redPin, GPIO.LOW)
            GPIO.output(yellowPin, GPIO.LOW)
            GPIO.output(greenPin, GPIO.HIGH)
            print("정상")
            radio3.select()

        wall = ImageTk.PhotoImage(Image.open(img_file), master=view)
        wall_label = Label(view, image = wall) 
        wall_label.place(x = 50,y = 200) 
        wall_label.pack()
    
        view.mainloop()

# 전체 GUI 메인코드
class App:

    # 초기실행코드(카메라관련)
    def __init__(self, window, window_title, video_source=-1):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = MyVideoCapture(self.video_source)

        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        
        self.btn_run=tkinter.Button(window, text="Run", width=50, command=self.run)
        self.btn_run.pack(anchor=tkinter.CENTER, expand=True)

        self.delay = 15
        self.update()

        self.window.mainloop()

    # 사진 촬영 후 스냅샷 저장
    def snapshot(self):
        ret, frame = self.vid.get_frame()
        frame = cv2.flip(frame, 1)

        if ret:
            global img_file
            img_file = ("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg")
            cv2.imwrite(img_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    # 비디오영상 업데이트용 함수
    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
         self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
         self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)
        
    # 딥러닝 예측 부분
    def run(self): 
        print('running....')
        img_pre = []
        img_pre.append(read_and_preprocess(img_file))
        prediction = model.predict(img_pre[0])
        global num
        num = prediction.argmax()
        print(num)

        # 최종결과화면 출력
        viewopen()

# 비디오 캡쳐용 코드
class MyVideoCapture:
    # 초기실행함수, 카메라 작동
    def __init__(self, video_source=-1):
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        
        if not self.vid.isOpened():
         raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    # 비디오 프레임 설정
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    # 종료 함수
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
# 메인 실행
App(tkinter.Tk(), "Tkinter and OpenCV")