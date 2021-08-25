import cv2
import imutils
import numpy as np
from PyQt5 import QtGui, QtWidgets
from keras.models import load_model
from keras.preprocessing.image import img_to_array


class Emotion_Rec:
    def __init__(self, model_path=None):

        # Parameters for loading data and images
        detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

        if model_path == None: # If no path is specified, the default model is used
            emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
        else:
            emotion_model_path=model_path


        # Load face detection model
        self.face_detection = cv2.CascadeClassifier(detection_model_path)  # Loading face detection model cascade classifier

        # Load face expression recognition model
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        # Expression Category
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]

    def run(self, frame_in,label_face):
        # frame_in Camera screen or image
        # canvas Background image for display
        # label_face Label object for face display screen
        # label_result The label object used to display the results

        # Adjust the screen size
        frame = imutils.resize(frame_in, width=300)  # Zoom screen
        # frame = cv2.resize(frame, (300,300))  # Zoom screen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion to grayscale

        # Detecting faces
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        preds = [] # Predicted results
        label = None # Predicted Tags
        (fX, fY, fW, fH) = None,None,None,None # Face position
        if len(faces) > 0:
            # Select the face with the largest ROI detected
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            # Extract the region of interest (ROI) from the grayscale map, convert its size to 64*64 pixels, and prepare the ROI for the classifier via CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Use the model to predict the probability of each classification
            preds = self.emotion_classifier.predict(roi)[0]
            # emotion_probability = np.max(preds)  # Maximum probability
            label = self.EMOTIONS[preds.argmax()]  # Select the expression class with the highest probability



        frameClone = frame.copy() 
        # canvas = 255* np.ones((250, 300, 3), dtype="uint8")
        # xu = cv2.imread('slice.png', flags=cv2.IMREAD_UNCHANGED)
        xlabel = []

        for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
            # # For displaying the probability of each category
            # text = "{}: {:.2f}%".format(emotion, prob * 100)
            xlabel.append([emotion,prob])
            # Circle the face area and display the recognition result
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0), 1)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)

        # # Resize the screen to fit the interface
        frameClone = cv2.resize(frameClone,(420,280))

        # Showing faces in the Qt interface
        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # QtWidgets.QApplication.processEvents()
        #
        # # Display the results in the label that shows the results
        # show = cv2.cvtColors(canva, cv2.COLOR_BGR2RGB)
        # showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        # label_result.setPixmap(QtGui.QPixmap.fromImage(showImage))

        return(label,xlabel)
