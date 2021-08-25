import sys,os
# load lib
from PyQt5.QtWidgets import *
from PyQt5 import sip
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import cv2
import time
from os import getcwd
import numpy as np
from real_time_video_me import Emotion_Rec
###draw image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")  # Declaration of use of QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from scipy import optimize
plt.rcParams['font.sans-serif'] = ['SimHei'] # Step 1 (replace sans-serif font)
plt.rcParams['axes.unicode_minus'] = False   # Step 2 (Solve the problem of displaying negative signs for negative numbers on the coordinate axes)
import io
import random
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import cv2
import threading
from random import random
import speech_recognition as sr
import pyaudio
from scipy.interpolate import make_interp_spline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import wave
import joblib
from xgboost import XGBClassifier
import librosa
import pandas as pd
#Importing a library of finished interfaces
from untitled import Ui_MainWindow

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        #Inheritance of (QMainWindow,Ui_MainWindow) parent class properties
        super(MainWindow,self).__init__()
        ##Initialisation of interface components
        self.setupUi(self)
        self.model_path = "models//_mini_XCEPTION.102-0.66.hdf5"  # model path
        self.timer_camera = QTimer()  # Timer
        self.cap = cv2.VideoCapture()  # get camera
        self.CAM_NUM = 0  # number of camera
        self.timer_camera1 = QTimer()  # timer
        self.label.setScaledContents(True)
        self.id = 1
        #function1
        """load image"""
        self.pushButton.clicked.connect(self.choose_pic)
        """switch on camera"""
        self.pushButton_2.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        """load video"""
        self.pushButton_5.clicked.connect(self.showVideo)
        self.timer_camera1.timeout.connect(self.show_camera1)
        #function2
        """semantic recognition"""
        self.pushButton_4.clicked.connect(self.talkRec)

        #function3
        """audio recognition"""
        self.pushButton_3.clicked.connect(self.talkRec2)
    def show_camera1(self):
        while (True):
            if self.id == 1:
                ref, frame = self.capture.read()
                # change form，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                canvas = cv2.imread('slice.jpg')  # use to display background
                canvas = cv2.resize(canvas, (380, 200))
                # count then start predict
                QApplication.processEvents()
                time_start = time.time()
                result,xlabel  = self.emotion_model.run(frame,self.label)
                time_end = time.time()
                # display results
                self.label_6.setText(result)
                self.label_5.setText(str(round((time_end - time_start), 3)) + ' s')
                if xlabel!= []:
                    try:
                        sip.delete(self.canvas1)
                        sip.delete(self.layout1)
                    except:
                        pass
                    self.fig1 = plt.Figure()
                    self.canvas1 = FC(self.fig1)
                    self.layout1 = QVBoxLayout()
                    self.layout1.addWidget(self.canvas1)
                    self.widget.setLayout(self.layout1)
                    ax = self.fig1.add_subplot(111)
                    if self.i == 20:
                        self.i = -1
                        self.x = []
                        self.y1 = []
                        self.y2 = []
                        self.y3 = []
                        self.y4 = []
                        self.y5 = []
                        self.y6 = []
                        self.y7 = []
                    self.i += 1
                    self.x.append(self.i)
                    self.y1.append(float(xlabel[0][1]))
                    self.y2.append(float(xlabel[1][1]))
                    self.y3.append(float(xlabel[2][1]))
                    self.y4.append(float(xlabel[3][1]))
                    self.y5.append(float(xlabel[4][1]))
                    self.y6.append(float(xlabel[5][1]))
                    self.y7.append(float(xlabel[6][1]))
                    ax.plot(self.x, self.y1, label=xlabel[0][0])
                    ax.plot(self.x, self.y2, label=xlabel[1][0])
                    ax.plot(self.x, self.y3, label=xlabel[2][0])
                    ax.plot(self.x, self.y4, label=xlabel[3][0])
                    ax.plot(self.x, self.y5, label=xlabel[4][0])
                    ax.plot(self.x, self.y6, label=xlabel[5][0])
                    ax.plot(self.x, self.y7, label=xlabel[6][0])
                    ax.legend()
                    self.canvas1.draw_idle()
                    self.canvas1.draw()  # TODO:start draw

                    try:
                        sip.delete(self.canvas5)
                        sip.delete(self.layout5)
                    except:
                        pass
                    self.fig5 = plt.Figure()
                    self.canvas5 = FC(self.fig5)
                    self.layout5 = QVBoxLayout()
                    self.layout5.addWidget(self.canvas5)
                    self.widget_4.setLayout(self.layout5)
                    ax = self.fig5.add_subplot(111)
                    x = [xlabel[0][0], xlabel[1][0], xlabel[2][0], xlabel[3][0], xlabel[4][0], xlabel[5][0], xlabel[6][0]]
                    y = [float(xlabel[0][1]), float(xlabel[1][1]), float(xlabel[2][1]), float(xlabel[3][1]),
                         float(xlabel[4][1]), float(xlabel[5][1]), float(xlabel[6][1])]
                    ax.bar(x, y)

                    self.canvas5.draw_idle()
                    self.canvas5.draw()  # TODO:start draw
            else:
                break
    def showVideo(self):
        # Use the file selection dialog to select a picture
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self, "Choose Video",
            "./",  # init path
            "Video(*.mp4)")  # file type
        if self.timer_camera.isActive():
            self.timer_camera.stop()  # turn off camera then start count
        # display log
        if fileName_choose != '':

            self.emotion_model = Emotion_Rec(self.model_path)
            self.capture = cv2.VideoCapture(fileName_choose)
            self.timer_camera1.start(30)
            self.id = 1
            try:
                sip.delete(self.canvas1)
                sip.delete(self.layout1)
            except:
                pass
            # Initialization Dynamic graph parameters
            self.i = -1
            self.x = []
            self.y1 = []
            self.y2 = []
            self.y3 = []
            self.y4 = []
            self.y5 = []
            self.y6 = []
            self.y7 = []
        else:
            self.label.setText('Video not selected')

        QApplication.processEvents()
    def talkRec(self):
        self.pushButton_4.setText("Recoding...")

        # semantic recognition
        def speech_text():
            init_rec = sr.Recognizer()
            print("Let's speak!!")
            with sr.Microphone() as source:
                init_rec.adjust_for_ambient_noise(source)
                audio_data = init_rec.listen(source)
                try:
                    text = init_rec.recognize_google(audio_data)
                except:
                    return ""
                else:
                    return text

      
        def bar_chart():
            global bar_c
            sia = SentimentIntensityAnalyzer()
            text = speech_text()
            data = sia.polarity_scores(text)
            return  data,text

            # {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

        data, text = bar_chart()
        print(text)
        self.pushButton_4.setText("Semantic Recognition")
        self.textEdit.setText(text)
        x = ['neg','neu','pos','compound']
        try:
            sip.delete(self.canvas2)
            sip.delete(self.layout2)
        except:
            pass
        self.fig2 = plt.Figure()
        self.canvas2 = FC(self.fig2)
        self.layout2 = QVBoxLayout()
        self.layout2.addWidget(self.canvas2)
        self.widget_3.setLayout(self.layout2)
        ax = self.fig2.add_subplot(111)
        ax.bar(x,[data[x[0]],data[x[1]],data[x[2]],data[x[3]]],color=['r','g','b','r'])
        self.canvas2.draw_idle()
        self.canvas2.draw()  # TODO:start draw

    def talkRec2(self):
        self.pushButton_3.setText("Recoding")
        # input wav file to extract features
        def extract_features(path):
            data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

            a_result = np.array([])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
            a_result = np.hstack((a_result, zcr))

            stft = np.abs(librosa.stft(data))
            chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            a_result = np.hstack((a_result, chroma_stft))

            mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
            a_result = np.hstack((a_result, mfcc))

            rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
            a_result = np.hstack((a_result, rms))

            mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
            a_result = np.hstack((a_result, mel))
            a_result = np.array(a_result)
            return a_result

        # record audio then predict
        def predict_audio():
            clf = joblib.load('xgboost.pkl')
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 5
            audio = pyaudio.PyAudio()

            # start Recording
            stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, input=True,
                                frames_per_buffer=CHUNK)
            print("recording...")
            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            print("finished recording")

            # stop Recording
            stream.stop_stream()
            stream.close()
            audio.terminate()

            waveFile = wave.open("demo.wav", 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

            test = extract_features("demo.wav")
            test = pd.DataFrame(test)
            test = test.T
            y_pred = clf.predict_proba(test)
            # sort
            # anger     disgust     scared   happy   sad  surprised  neutral
            return y_pred
            # [[0.5192316  0.04025906 0.2138472  0.17755078 0.02496348 0.00798158
        # 0.01616629]]
        y_pred = predict_audio()
        self.pushButton_3.setText("Audio Recognition")
        try:
            sip.delete(self.canvas3)
            sip.delete(self.layout3)
        except:
            pass
        self.fig3 = plt.Figure()
        self.canvas3 = FC(self.fig3)
        self.layout3 = QVBoxLayout()
        self.layout3.addWidget(self.canvas3)
        self.widget_2.setLayout(self.layout3)
        ax = self.fig3.add_subplot(111)
        ax.bar(['anger','disgust','scared','happy','sad','surprised','neutral'],[y_pred[0][0],y_pred[0][1],y_pred[0][2],y_pred[0][3],y_pred[0][4],y_pred[0][5],y_pred[0][6]],color=['r','g','b','r','g','b','c'])
        self.canvas3.draw_idle()
        self.canvas3.draw()  # TODO:draw

    def choose_pic(self):
        # Use the file selection dialog to select a picture
        fileName_choose, filetype = QFileDialog.getOpenFileName(
                                self, "Select image file",
                                "./",  
                                "Image(*.jpg;*.jpeg;*.png)") 
        self.path = fileName_choose
        self.timer_camera.stop() 
       
        if fileName_choose != '':
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            if self.timer_camera1.isActive():
                self.timer_camera1.stop()
            self.label.clear()
            self.id = 3
            self.emotion_model = Emotion_Rec(self.model_path)
            image = self.cv_imread(fileName_choose) # load file
            # count and predict
            QApplication.processEvents()
            time_start = time.time()
            result,xlabel  = self.emotion_model.run(image,self.label)
            # self.widget_4...
            time_end = time.time()

            try:
                sip.delete(self.canvas5)
                sip.delete(self.layout5)
            except:
                pass
            self.fig5 = plt.Figure()
            self.canvas5 = FC(self.fig5)
            self.layout5 = QVBoxLayout()
            self.layout5.addWidget(self.canvas5)
            self.widget_4.setLayout(self.layout5)
            ax = self.fig5.add_subplot(111)
            x = [xlabel[0][0],xlabel[1][0],xlabel[2][0],xlabel[3][0],xlabel[4][0],xlabel[5][0],xlabel[6][0]]
            y = [float(xlabel[0][1]),float(xlabel[1][1]),float(xlabel[2][1]),float(xlabel[3][1]),float(xlabel[4][1]),float(xlabel[5][1]),float(xlabel[6][1])]
            ax.bar(x,y)

            self.canvas5.draw_idle()
            self.canvas5.draw()  # TODO:draw

            # show results
            self.label_6.setText(result)
            self.label_5.setText(str(round((time_end - time_start), 3)) + ' s')
        else:
            self.label.setText('Img not selected')

        QApplication.processEvents()

    def cv_imread(self, filePath):
        # load image
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        ## imdecode reads rgb, if the subsequent need for opencv processing, you need to convert to bgr, the conversion of the image color will change
        ## cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img
    def button_open_camera_click(self):
        if self.timer_camera1.isActive():
            self.timer_camera1.stop()
            print("Camera off")
            self.id = 0
            self.label.clear()
            # MainWindow{border-image:url(back.jpg);}
        if self.timer_camera.isActive() == False: #Check timing status
            flag = self.cap.open(self.CAM_NUM) # Check camera status
            if flag == False:
                QMessageBox.warning(self, u"Warning",u"Please check if the camera is properly connected to the computer!",QMessageBox.Yes,QMessageBox.Yes)
            else:
                print("开始")
                try:
                    sip.delete(self.canvas1)
                    sip.delete(self.layout1)
                except:
                    pass
                self.label.setText('Identification system being activated...\n\nleading')
                self.emotion_model = Emotion_Rec(self.model_path)
                # Turn on the timer
                self.timer_camera.start(30)
                self.id = 0
                #Initialization Dynamic graph parameters
                self.i = -1
                self.x = []
                self.y1 = []
                self.y2 = []
                self.y3 = []
                self.y4 = []
                self.y5 = []
                self.y6 = []
                self.y7 = []

    def show_camera(self):
        if self.id ==0:
            print("fresh")
            # Timer slot function, executed at regular intervals
            flag, self.image = self.cap.read() # get frame
            self.image=cv2.flip(self.image, 1) # Left and right flip
            time_start = time.time() # count
            # Prediction using models
            result,xlabel = self.emotion_model.run(self.image,self.label)
            time_end = time.time()
            # Show results in the interface
            self.label_6.setText(result)
            self.label_5.setText(str(round((time_end-time_start),3))+' s')
            ####Dynamic Loading Line Chart
            if xlabel != []:
                try:
                    sip.delete(self.canvas1)
                    sip.delete(self.layout1)
                except:
                    pass
                self.fig1 = plt.Figure()
                self.canvas1 = FC(self.fig1)
                self.layout1 = QVBoxLayout()
                self.layout1.addWidget(self.canvas1)
                self.widget.setLayout(self.layout1)
                ax = self.fig1.add_subplot(111)
                if self.i == 20:
                    self.i = -1
                    self.x = []
                    self.y1 = []
                    self.y2 = []
                    self.y3 = []
                    self.y4 = []
                    self.y5 = []
                    self.y6 = []
                    self.y7 = []
                self.i+=1

                self.x.append(self.i)

                self.y1.append(float(xlabel[0][1]))
                self.y2.append(float(xlabel[1][1]))
                self.y3.append(float(xlabel[2][1]))
                self.y4.append(float(xlabel[3][1]))
                self.y5.append(float(xlabel[4][1]))
                self.y6.append(float(xlabel[5][1]))
                self.y7.append(float(xlabel[6][1]))
                ax.plot(self.x, self.y1, label=xlabel[0][0])
                ax.plot(self.x, self.y2, label=xlabel[1][0])
                ax.plot(self.x, self.y3, label=xlabel[2][0])
                ax.plot(self.x, self.y4, label=xlabel[3][0])
                ax.plot(self.x, self.y5, label=xlabel[4][0])
                ax.plot(self.x, self.y6, label=xlabel[5][0])
                ax.plot(self.x, self.y7, label=xlabel[6][0])
                ax.legend()
                self.canvas1.draw_idle()
                self.canvas1.draw()  # TODO:draw

                try:
                    sip.delete(self.canvas5)
                    sip.delete(self.layout5)
                except:
                    pass
                self.fig5 = plt.Figure()
                self.canvas5 = FC(self.fig5)
                self.layout5 = QVBoxLayout()
                self.layout5.addWidget(self.canvas5)
                self.widget_4.setLayout(self.layout5)
                ax = self.fig5.add_subplot(111)
                x = [xlabel[0][0], xlabel[1][0], xlabel[2][0], xlabel[3][0], xlabel[4][0], xlabel[5][0], xlabel[6][0]]
                y = [float(xlabel[0][1]), float(xlabel[1][1]), float(xlabel[2][1]), float(xlabel[3][1]),
                     float(xlabel[4][1]), float(xlabel[5][1]), float(xlabel[6][1])]
                ax.bar(x, y)

                self.canvas5.draw_idle()
                self.canvas5.draw()  # TODO:draw

if __name__ == "__main__":
    #Create QApplication Fixed Write
    app = QApplication(sys.argv)
    # Instantiation interface
    window = MainWindow()
    #Display Interface
    window.show()
    #Blocking, fixed writing
    sys.exit(app.exec_())