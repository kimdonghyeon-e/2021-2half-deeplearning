from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from matplotlib.pyplot import imread
import imageio #pip imstall imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import cv2
import tensorflow as tf
from PIL import ImageOps, Image
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#GPIO 불러오는 코드
import Jetson.GPIO as GPIO
import time

#이미지 사이즈 조절용 함수
def read_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) # reading the image
    img = cv2.resize(img, (256, 256)) # resizing it (I just like it to be powers of 2)
    img = np.array(img, dtype='float32') # convert its datatype so that it could be normalized
    img = img/255 # normalization (now every pixel is in the range of 0 and 1)
    img = np.expand_dims(img, axis=0)
    return img

opt = optimizers.Adam(learning_rate=0.0001)

# Load pretrained model (best saved one)
with open('covid_classifier_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
# load the model  
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('covid_classifier_weights.h5')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

root = Tk()
root.title('covid check_select image')
root.geometry('700x700')
 
def open():         #파일 가져올부분
    global my_image # 함수에서 이미지를 기억하도록 전역변수 선언 (안하면 사진이 안보임)
    root.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
    ('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))

    print(root.filename)
    global openedimg
    openedimg = imread(root.filename)

    imageio.imwrite('saved.png', openedimg)

    showimg = imread('saved.png')
    showimg = cv2.resize(showimg, (500, 500))
    imageio.imwrite('sh.png', showimg)

    Label(root, text=root.filename).pack() # 파일경로 view
    my_image = ImageTk.PhotoImage(Image.open('sh.png'))
    # my_image = cv2.resize(my_image, (450, 450))
    Label(image=my_image).pack() #사진 view
    btn_submit = Button(root, text='검사', command=run)
    btn_submit.pack()

def run():          #딥러닝 돌릴부분(만들기)
    print('hi')
    root.destroy()
    img_pre = []
    img_pre.append(read_and_preprocess('./saved.png'))
    prediction = model.predict(img_pre[0])
    global num
    num = prediction.argmax()
    print(num)
    viewopen()
    
def viewopen():     #돌린거 보여줄부분
    view = Tk()
    view.title('covid check_view image')
    view.geometry('700x700')
    chkvar = IntVar() # chkvar 에 int 형으로 값을 저장한다
    chkbox = Checkbutton(view, text="코로나 확진", variable=chkvar)
    # chkbox.select() # 자동 선택 처리
    # chkbox.deselect() # 선택 해제 처리
    # chkbox.pack()

    chkvar2 = IntVar()
    chkbox2 = Checkbutton(view, text="일반 폐렴", variable=chkvar2)
    # chkbox2.pack()
    #chkbox2.select() # 자동 선택 처리
    # chkbox2.deselect() # 선택 해제 처리

    chkvar3 = IntVar()
    chkbox3 = Checkbutton(view, text="정상", variable=chkvar3)
    # chkbox3.pack()
    #chkbox3.select() # 자동 선택 처리
    #chkbox3.deselect() # 선택 해제 처리

    print(num)

    #결과를 체크박스로 표시
    if num == 0:
        chkbox.select()
    elif num == 1:
        chkbox2.select()
    else:
        chkbox3.select()

    chkbox.pack()
    chkbox2.pack()
    chkbox3.pack()

    wall = ImageTk.PhotoImage(Image.open("sh.png"), master=view)
    wall_label = Label(view, image = wall) 
    wall_label.place(x = 50,y = 200) 
    wall_label.pack()

    #num 값에 따라 LED 점등
    if num == 0:
        GPIO.output(redPin, GPIO.HIGH)
        GPIO.output(yellowPin, GPIO.LOW)
        GPIO.output(greenPin, GPIO.LOW)
        #print("우한폐렴")
        time.sleep(2)
        
    elif num == 1:
        GPIO.output(redPin, GPIO.LOW)
        GPIO.output(yellowPin, GPIO.HIGH)
        GPIO.output(greenPin, GPIO.LOW)
        #print("일반폐렴")
        time.sleep(2)

    elif num == 2:
        GPIO.output(redPin, GPIO.LOW)
        GPIO.output(yellowPin, GPIO.LOW)
        GPIO.output(greenPin, GPIO.HIGH)
        #print("정상")
        time.sleep(2)
    else:
        GPIO.output(redPin, GPIO.LOW)
        GPIO.output(yellowPin, GPIO.LOW)
        GPIO.output(greenPin, GPIO.LOW)
        time.sleep(2)
    #GPIO 끝 

    view.mainloop()

#GPIO 출력
#LED 핀 지정

redPin = 11
yellowPin = 12
greenPin = 13

GPIO.setmode(GPIO.BOARD)

GPIO.setup(redPin, GPIO.OUT)
GPIO.setup(yellowPin, GPIO.OUT)
GPIO.setup(greenPin, GPIO.OUT)

GPIO.output(redPin, GPIO.LOW)
GPIO.output(yellowPin, GPIO.LOW)
GPIO.output(greenPin, GPIO.LOW)

my_btn = Button(root, text='파일열기', command=open)
my_btn.pack()

root.mainloop()