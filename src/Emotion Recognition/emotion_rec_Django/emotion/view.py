import io
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import numpy as np
from django.http import HttpResponse, request
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from random import random
import speech_recognition as sr
import pyaudio
from scipy.interpolate import make_interp_spline
from tensorflow.python.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recog_original as sro

@gzip.gzip_page
def Home(request):
    global init_c
    init_c = init_chart()
    return render(request, 'app1.html')


# return your camera
def frame_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")


# return line chart with emotion recognition
def data1(request):
    return StreamingHttpResponse(data(), content_type="multipart/x-mixed-replace;boundary=frame")

def f_bar_chart(request):

    return StreamingHttpResponse(f_bar_data(), content_type="multipart/x-mixed-replace;boundary=frame")


def text_bar_chart(request):
    return StreamingHttpResponse(bar(), content_type="multipart/x-mixed-replace;boundary=frame")

def audio_bar_stream(request):
    return StreamingHttpResponse(audio_bar(), content_type="multipart/x-mixed-replace;boundary=frame")

def init_chart():
    fig5 = plt.figure(4)
    ax5 = fig5.add_subplot(111)
    buf5 = io.BytesIO()
    fig5.savefig(buf5, format='png')
    ax5.remove()
    buf5.seek(0)
    file_bytes5 = np.asarray(bytearray(buf5.read()), dtype=np.uint8)
    chart5 = cv2.imdecode(file_bytes5, cv2.IMREAD_COLOR)
    chart5 = cv2.resize(chart5, (480, 240))
    _, jpeg5 = cv2.imencode('.jpg', chart5)
    return jpeg5

a_b_tem = init_chart()
def audio_bar_c():
    global a_b_tem
    emotion =["anger",  "disgust",  "scared","happy",  "neutral", "sad", "surprise"]
    prob = sro.predict_audio()
    print(prob)
    fig6 = plt.figure(6)
    ax = fig6.add_subplot(111)
    ax.bar(emotion, prob, color=["orange", "green", "red", "m", "gray", "cyan", "pink"], width=0.4)
    # plt.legend()
    ax.set_xlabel('Emotion')
    ax.set_ylabel('rate')
    ax.set(ylim=(0, 1))
    buf1 = io.BytesIO()
    fig6.savefig(buf1, format='png')
    ax.remove()
    buf1.seek(0)
    file_bytes = np.asarray(bytearray(buf1.read()), dtype=np.uint8)
    chart1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    chart1 = cv2.resize(chart1, (480, 240))
    _, jpeg = cv2.imencode('.jpg', chart1)
    a_b_tem = jpeg
    return a_b_tem.tobytes()
# ------------------------------speech to text emotion analysis ------------------------------------
def speech_text():
    init_rec = sr.Recognizer()
    print("Let's speak!!")
    #
    # while True:
    with sr.Microphone() as source:
        init_rec.adjust_for_ambient_noise(source)
        audio_data = init_rec.listen(source)
        try:
            text = init_rec.recognize_google(audio_data)
        except:
            return ""
        else:
            return text


t_b_tem = init_chart()
def bar_chart():
    global bar_c, t_b_tem
    sia = SentimentIntensityAnalyzer()
    text = speech_text()
    print(text)
    data = sia.polarity_scores(text)
    name = list(data.keys())
    values = list(data.values())
    print(data)
    if (values[0] != 0 or values[1] != 0 or values[2] != 0):
        fig2 = plt.figure(3)
        ax = fig2.add_subplot(111)
        ax.bar(name, values, color=['darkgrey', "cyan", "green", "orange"], width=0.4)
        # plt.legend()
        ax.set_xlabel('Emotion')
        ax.set_ylabel('rate')
        ax.set(ylim=(-1, 1))
        buf1 = io.BytesIO()
        fig2.savefig(buf1, format='png')
        ax.remove()
        buf1.seek(0)
        file_bytes = np.asarray(bytearray(buf1.read()), dtype=np.uint8)
        chart1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        chart1 = cv2.resize(chart1, (480, 240))
        _, jpeg = cv2.imencode('.jpg', chart1)
        t_b_tem = jpeg
    return t_b_tem.tobytes()
    # return HttpResponse(buf1,content_type="image/png")


def bar():
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bar_chart() + b'\r\n\r\n')

def audio_bar():
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + audio_bar_c() + b'\r\n\r\n')


y = []
q = []
x1 = 0
x = []
a = 10
b = 0

def curve(x,y):
    if(len(x) > 3):
        npx = np.array(x)
        npy = np.array(y)
        newx = np.linspace(npx.min(), npx.max(), 200)
        spl = make_interp_spline(npx, npy, k=3)
        newy = spl(newx)
    else:
        newx = x
        newy = y
    return [newx, newy]
# post a chart image to a url
# ----------------------line chart for emotion recognition-----------------------------------------


y1, y2, y3, y4, y5, y6, y7 = [], [], [], [], [], [], []
def chart():
    global x1, a, b
    if len(all_pred)>0:
        y_arr = [y1, y2, y3, y4, y5, y6, y7]
        for y, n in zip(y_arr, range(7)):
            y.append(all_pred[n])
        x1 += 1
        x.append(x1)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        if (len(y1) > 10):
            for y in y_arr:
                y.pop(0)
            x.pop(0)
            a += 1
            b += 1
        color = ["orange", "green", "red", "m", "cyan", "pink", "gray"]
        for y, colors in zip(y_arr,color):
            ax.plot(curve(x, y)[0], curve(x, y)[1], color=colors, linewidth=1)
        ax.axis([b, a, 0, 1])
        ax.set_xlabel('iteration times')
        ax.set_ylabel('rate')

        # save chart as image
        # fig1 = plt.gcf()
        # plt.close()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        ax.remove()
        buf.seek(0)
        file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
        chart = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        chart = cv2.resize(chart, (480, 240))
        if (len(facetem) > 0):
            facecrop1 = cv2.resize(facetem[0], (100, 100))
            newface = cv2.copyMakeBorder(facecrop1, 70, 70, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            img1 = cv2.hconcat([newface, chart])
            _, jpeg1 = cv2.imencode('.jpg', img1)
        else:
            _, jpeg1 = cv2.imencode('.jpg', chart)
        # return HttpResponse(jpeg1.tobytes(),content_type="image/png")
        return jpeg1.tobytes()
    else:
        return init_c.tobytes()




def f_bar():
    if len(all_pred)>0:
        x = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        y = [all_pred[0], all_pred[1], all_pred[2], all_pred[3], all_pred[4], all_pred[5], all_pred[6]]
        fig1 = plt.figure(2)
        ax = fig1.add_subplot(111)
        ax.bar(x, y, color=["orange", "green", "red", "m", "cyan", "pink", "gray"], width=0.4)
        ax.set(ylim=(0, 1))
        ax.set_xlabel('emotion')
        ax.set_ylabel('rate')
        buf2 = io.BytesIO()
        fig1.savefig(buf2, format='png')
        ax.remove()
        buf2.seek(0)
        file_bytes = np.asarray(bytearray(buf2.read()), dtype=np.uint8)
        chart = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        chart = cv2.resize(chart, (480, 240))
        if (len(facetem) > 0):
            facecrop1 = cv2.resize(facetem[0], (100, 100))
            newface = cv2.copyMakeBorder(facecrop1, 70, 70, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            img1 = cv2.hconcat([newface, chart])
            _, jpeg1 = cv2.imencode('.jpg', img1)
        else:
            _, jpeg1 = cv2.imencode('.jpg', chart)
        return jpeg1.tobytes()
    else:
        return init_c.tobytes()



def data():

    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + chart() + b'\r\n\r\n')


def f_bar_data():

    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + f_bar() + b'\r\n\r\n')


# ---------------------------------end--------------------------------------------------------------------------

# ---------------------------face detection using carmera--------------------------------------------
facetem = []
all_pred = []
# detect face and crop it
#process iamge
def face(img):
    global facetem,all_pred
    facea = []
    face_orig = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if (len(faces_detected) > 0):
        for (x, y, w, h) in faces_detected:
            face_orig = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            facecrop = img[y:y + h, x:x + w]
            all_pred = emotion_pred(facecrop)
            facea.append(facecrop) #append face image to array

    facetem = facea #save to global variable

    return face_orig #return processed image



def emotion_pred(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = emotion_classifier.predict(face)[0]
    return preds

# to capture video class
#get camera
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    #get each frame from camera
    def get_frame(self):
        image = self.frame
        #get frame after process
        jpeg = face(image)
        _, jpeg1 = cv2.imencode('.jpg', jpeg)
        return jpeg1.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    global emotion_classifier
    emotion_model_path = 'fer_sm_M.124-0.74.hdf5'
    emotion_classifier = load_model(emotion_model_path, compile=False)

    while True:
        frame = camera.get_frame() #post to streamhttpresponse
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# -----------------------------------------------------------------------------------------------------
