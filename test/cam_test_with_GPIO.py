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
from PIL import ImageTk, Image
import threading 
import time

#GPIO 불러오는 코드
import Jetson.GPIO as GPIO

opt = optimizers.Adam(learning_rate=0.0001)
# Load pretrained model (best saved one)
with open('covid_classifier_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
# load the model  
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('covid_classifier_weights.h5')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

#이미지 사이즈 조절용 함수
def read_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) # reading the image
    img = cv2.resize(img, (256, 256)) # resizing it (I just like it to be powers of 2)
    img = np.array(img, dtype='float32') # convert its datatype so that it could be normalized
    img = img/255 # normalization (now every pixel is in the range of 0 and 1)
    img = np.expand_dims(img, axis=0)
    return img

def viewopen():          #딥러닝 돌릴부분(만들기)
        view = Tk()
        view.title('covid check_view image')
        view.geometry('700x700')
        chkvar = IntVar() # chkvar 에 int 형으로 값을 저장한다
        radio1 = tkinter.Radiobutton(view, text="코로나 확진", variable=chkvar, value=0)
        # chkbox.deselect() # 선택 해제 처리
        radio1.pack()

        chkvar2 = IntVar()
        radio2 = tkinter.Radiobutton(view, text="일반 폐렴", variable=chkvar2, value=1)
        radio2.pack()
        #chkbox2.select() # 자동 선택 처리
        # chkbox2.deselect() # 선택 해제 처리

        chkvar3 = IntVar()
        radio3 = tkinter.Radiobutton(view, text="정상", variable=chkvar3, value=2)
        radio3.pack()
        #chkbox3.select() # 자동 선택 처리
        #chkbox3.deselect() # 선택 해제 처리
        
        if(num==0):
            radio1.select()
        elif(num==1):
            radio2.select()
        else:
            radio3.select()

        wall = ImageTk.PhotoImage(Image.open(img_file), master=view)
        wall_label = Label(view, image = wall) 
        wall_label.place(x = 50,y = 200) 
        wall_label.pack()
    
        view.mainloop()

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        
        # Button that lets the user take a run
        self.btn_run=tkinter.Button(window, text="Run", width=50, command=self.run)
        self.btn_run.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        frame = cv2.flip(frame, 1)

        if ret:
            global img_file
            img_file = ("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg")
            cv2.imwrite(img_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
         self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
         self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)
        
        
    def run(self):          #딥러닝 돌릴부분(만들기)
        print('running....')
        #self.destroy()
        img_pre = []
        img_pre.append(read_and_preprocess(img_file))
        prediction = model.predict(img_pre[0])
        global num
        num = prediction.argmax()
        print(num)
        viewopen()

 
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        
        if not self.vid.isOpened():
         raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
     # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")

#LED 핀 지정
redPin = 11
yellowPin = 12
greenPin = 13

GPIO.setmode(GPIO.BOARD)

GPIO.setup(redPin, GPIO.OUT)
GPIO.setup(yellowPin, GPIO.OUT)
GPIO.setup(greenPin, GPIO.OUT)


#num 값에 따라 LED 점등
if num == 0:
    GPIO.output(redPin, GPIO.HIGH)
    GPIO.output(yellowPin, GPIO.LOW)
    GPIO.output(greenPin, GPIO.LOW)
    print("우한폐렴")
    time.sleep(2)
    
elif num == 1:
    GPIO.output(redPin, GPIO.LOW)
    GPIO.output(yellowPin, GPIO.HIGH)
    GPIO.output(greenPin, GPIO.LOW)
    print("일반폐렴")
    time.sleep(2)

elif num == 2:
    GPIO.output(redPin, GPIO.LOW)
    GPIO.output(yellowPin, GPIO.LOW)
    GPIO.output(greenPin, GPIO.HIGH)
    print("정상")
    time.sleep(2)
else:
    GPIO.output(redPin, GPIO.LOW)
    GPIO.output(yellowPin, GPIO.LOW)
    GPIO.output(greenPin, GPIO.LOW)
    time.sleep(2)
