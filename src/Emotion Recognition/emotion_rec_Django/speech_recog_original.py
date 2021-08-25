import io
import random
from scipy import interpolate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
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


# 语言识别返回文本
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


# 初始化一张空白图片
img = np.zeros([100, 100, 3], dtype=np.uint8)
img.fill(255)  # or img[:] = 255
_, jpeg = cv2.imencode('.jpg', img)
bar_c = jpeg


# 目前用cv2显示图片， 但是可以用变量jpeg返回图片
def bar_chart():
    global bar_c
    sia = SentimentIntensityAnalyzer()
    text = speech_text()
    print(text)
    data = sia.polarity_scores(text)
    name = list(data.keys())
    values = list(data.values())
    fig2 = plt.figure(3)
    ax = fig2.add_subplot(111)
    print(data)
    if (values[0] != 0 or values[1] != 0 or values[2] != 0):
        ax.bar(name, values, color='darkgrey', width=0.4)
        # plt.legend()
        ax.set_xlabel('Emotion')
        ax.set_ylabel('rate')
        buf1 = io.BytesIO()
        fig2.savefig(buf1, format='png')
        buf1.seek(0)
        file_bytes = np.asarray(bytearray(buf1.read()), dtype=np.uint8)
        chart1 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        chart1 = cv2.resize(chart1, (480, 240))
        cv2.imshow('audio', chart1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #     _, jpeg = cv2.imencode('.jpg', chart1)
    #     bar_c = jpeg
    # ax.remove()
    # return bar_c


# 输入wav文件的路径，提取特征
def extract_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    #     chroma_cq = np.mean(librosa.feature.chroma_cqt(y=data, sr=sample_rate).T, axis=0)
    #     result = np.hstack((result, chroma_cq)) # stacking horizontally

    #     chroma_cens = np.mean(librosa.feature.chroma_cens(y=data, sr=sample_rate).T, axis=0)
    #     result = np.hstack((result, chroma_cens))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    cent = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, cent))

    sb = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, sb))

    sc = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, sc))

    sf = np.mean(librosa.feature.spectral_flatness(y=data).T, axis=0)
    result = np.hstack((result, sf))

    sr = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate, roll_percent=0.95).T, axis=0)
    result = np.hstack((result, sr))

    sc = np.mean(librosa.feature.poly_features(S=stft, order=2).T, axis=0)
    result = np.hstack((result, sc))

    ton = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, ton))
    return np.array(result)


# 录制音频然后预测，最后返回7个情绪分类的概率
def predict_audio():
    clf = joblib.load("MLP.pkl")
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
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
    transfer=joblib.load('std_scaler.bin')
    test = transfer.transform(test)
    y_pred = clf.predict_proba(test)
    # y_pred = clf.predict(test)
    # 顺序应该是
    # anger     disgust     fear   happy    neutral sad surprise
    return y_pred[0]

