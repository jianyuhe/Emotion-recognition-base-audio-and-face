import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image 

dataset_path = 'fer2013.csv'
image_size=(48,48)

def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            # cv2.imshow('a', face)
            # cv2.waitKey(0)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        # emotions = pd.get_dummies(data['emotion']).as_matrix()
        emotions = pd.get_dummies(data['emotion']).values
        
        return faces, emotions
    
def load_affectnet(dir_path, num_class):
    images_path = os.path.join(dir_path, 'images')
    annotations_path = os.path.join(dir_path, 'annotations')
    faces = []
    labels = []
    num = 0
    for file in os.listdir(images_path):
        num = num + 1
        image = Image.open(os.path.join(images_path, file))
        image_arr = np.array(image)
        faces.append(image_arr)
        id = file.split('.')[0]
        label = np.load(os.path.join(annotations_path, id+'_exp.npy'))
        labels.append(np.eye(num_class)[int(label.tolist()[0])])

    faces = np.asarray(faces)
    labels = np.asarray(labels)
    return faces, labels
    

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def preprocess_input_0(x):
    x = x.astype('float32')
    mean = np.mean(x, axis=0)
    x = x - mean
    # std = np.std(x)
    # x = (x - mean) / std
    return x

