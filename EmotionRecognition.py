import sys
from PySide.QtCore import *
from PySide.QtGui import *
import os
import imageio
import numpy as np
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
import cv2
import h5py
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import load_model
from keras.preprocessing import image
import matplotlib
from matplotlib import pyplot as plt
from array import *
import time

class EmotionRecognition(QMainWindow):
    custom = None
    cam = 0
    img = 0
    i = 0
    flagCam = 0

    def __init__(self, parent=None):
        super(EmotionRecognition,self).__init__()
        loadUi('EmotionRecognition.ui',self)

        global i, cam
        i = 1
        cam = 0

        # Members.
        self.imageHolderScene = QGraphicsScene()
        self.neutralScene = QGraphicsScene()
        self.disgustScene = QGraphicsScene()
        self.angryScene = QGraphicsScene()
        self.sadScene = QGraphicsScene()
        self.fearScene = QGraphicsScene()
        self.surpriseScene = QGraphicsScene()
        self.happyScene = QGraphicsScene()
        self.capture = None

        # Functionality mapping UI to Script.
        self.imageHolder.setScene(self.imageHolderScene)
        self.neutral.setScene(self.neutralScene)
        self.disgust.setScene(self.disgustScene)
        self.angry.setScene(self.angryScene)
        self.sad.setScene(self.sadScene)
        self.fear.setScene(self.fearScene)
        self.surprise.setScene(self.surpriseScene)
        self.happy.setScene(self.happyScene)
       
        self.loadImage.clicked.connect(self.loadImageClicked)
        self.showGraph.clicked.connect(self.showGraphClicked)
        self.openCamera.clicked.connect(self.openCameraClicked)
        self.closeCamera.clicked.connect(self.closeCameraClicked)
        self.capturePhoto.clicked.connect(self.capturePhotoClicked)
        self.exit.clicked.connect(self.exitClicked)
        self.liveFeed.clicked.connect(self.liveFeedClicked)

        self.filepath = "Normal\\Neutral.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.neutralScene.clear()
        self.neutralScene.addPixmap(QPixmap(self.filepath))
        self.neutral.fitInView(self.neutralScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatioByExpanding)

        self.filepath = "Normal\\Disgust.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.disgustScene.clear()
        self.disgustScene.addPixmap(QPixmap(self.filepath))
        self.disgust.fitInView(self.disgustScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatioByExpanding)

        self.filepath = "Normal\\Angry.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.angryScene.clear()
        self.angryScene.addPixmap(QPixmap(self.filepath))
        self.angry.fitInView(self.angryScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatioByExpanding)

        self.filepath = "Normal\\Sad.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.sadScene.clear()
        self.sadScene.addPixmap(QPixmap(self.filepath))
        self.sad.fitInView(self.sadScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatioByExpanding)

        self.filepath = "Normal\\Fear.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.fearScene.clear()
        self.fearScene.addPixmap(QPixmap(self.filepath))
        self.fear.fitInView(self.fearScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatioByExpanding)

        self.filepath = "Normal\\Surprise.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.surpriseScene.clear()
        self.surpriseScene.addPixmap(QPixmap(self.filepath))
        self.surprise.fitInView(self.surpriseScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatioByExpanding)

        self.filepath = "Normal\\Happy.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.happyScene.clear()
        self.happyScene.addPixmap(QPixmap(self.filepath))
        self.happy.fitInView(self.happyScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatioByExpanding)

    def reset(self):
        self.status.setText("NULL")
        self.confidence.setText("NULL")

        self.filepath = "Normal\\Neutral.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.neutralScene.clear()
        self.neutralScene.addPixmap(QPixmap(self.filepath))
        self.neutral.fitInView(self.neutralScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        self.filepath = "Normal\\Disgust.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.disgustScene.clear()
        self.disgustScene.addPixmap(QPixmap(self.filepath))
        self.disgust.fitInView(self.disgustScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        self.filepath = "Normal\\Angry.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.angryScene.clear()
        self.angryScene.addPixmap(QPixmap(self.filepath))
        self.angry.fitInView(self.angryScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        self.filepath = "Normal\\Sad.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.sadScene.clear()
        self.sadScene.addPixmap(QPixmap(self.filepath))
        self.sad.fitInView(self.sadScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        self.filepath = "Normal\\Fear.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.fearScene.clear()
        self.fearScene.addPixmap(QPixmap(self.filepath))
        self.fear.fitInView(self.fearScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        self.filepath = "Normal\\Surprise.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.surpriseScene.clear()
        self.surpriseScene.addPixmap(QPixmap(self.filepath))
        self.surprise.fitInView(self.surpriseScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        self.filepath = "Normal\\Happy.png"
        self.image = imageio.imread(self.filepath)
        self.image = cv2.resize(self.image,(50,50))
        self.happyScene.clear()
        self.happyScene.addPixmap(QPixmap(self.filepath))
        self.happy.fitInView(self.happyScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def emotion_analysis(emotions):
        objects = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, emotions, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Percentage')
        plt.title('Emotion Analysis')
        plt.show()

    def loadImageClicked(self):
        # Get image file path.
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open 1st Image ...', filter="Picture Files (*.jpg *.png)")
        if not filePath: return

        EmotionRecognition.reset(self)
        self.showGraph.setDisabled(False)
        # Initialize scene.
        self.image = imageio.imread(filePath)
        self.image = cv2.resize(self.image,(48,48))

        self.imageHolderScene.clear()
        self.imageHolderScene.addPixmap(QPixmap(filePath))
        self.imageHolder.fitInView(self.imageHolderScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

        model = models.Sequential()

        #1st convolution layer
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
        model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
        #2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
        #3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
        model.add(Flatten())
 
        #fully connected neural networks
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
 
        model.add(Dense(7, activation='softmax'))

        model.load_weights('facial_expression_model_weights.h5')

        img = image.load_img(filePath, grayscale=True, target_size=(48, 48))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        custom = model.predict(img_tensor)
       
        maxindex = custom.argmax()
        print(maxindex)

        emotion_array = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        print(emotion_array[maxindex])

        a = custom[0][maxindex].astype(float)
        a = a*100
        a = np.round(a,2)
        
        EmotionRecognition.custom = custom[0]

        b = a.astype(np.str)

        stat = emotion_array[maxindex]

        self.status.setText(stat)
        self.confidence.setText(b+"%")
        
        if(stat == "Angry"):
            self.filepath = "Color\\Angry.png"
            self.image = imageio.imread(self.filepath)
            self.image = cv2.resize(self.image,(50,50))
            self.angryScene.clear()
            self.angryScene.addPixmap(QPixmap(self.filepath))
            self.angry.fitInView(self.angryScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif(stat == "Disgust"):
            self.filepath = "Color\\Disgust.png"
            self.image = imageio.imread(self.filepath)
            self.image = cv2.resize(self.image,(50,50))
            self.disgustScene.clear()
            self.disgustScene.addPixmap(QPixmap(self.filepath))
            self.disgust.fitInView(self.disgustScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif(stat == "Fear"):
            self.filepath = "Color\\Fear.png"
            self.image = imageio.imread(self.filepath)
            self.image = cv2.resize(self.image,(50,50))
            self.fearScene.clear()
            self.fearScene.addPixmap(QPixmap(self.filepath))
            self.fear.fitInView(self.fearScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif(stat == "Happy"):
            self.filepath = "Color\\Happy.png"
            self.image = imageio.imread(self.filepath)
            self.image = cv2.resize(self.image,(50,50))
            self.happyScene.clear()
            self.happyScene.addPixmap(QPixmap(self.filepath))
            self.happy.fitInView(self.happyScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif(stat == "Sad"):
            self.filepath = "Color\\Sad.png"
            self.image = imageio.imread(self.filepath)
            self.image = cv2.resize(self.image,(50,50))
            self.sadScene.clear()
            self.sadScene.addPixmap(QPixmap(self.filepath))
            self.sad.fitInView(self.sadScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif(stat == "Surprise"):
            self.filepath = "Color\\Surprise.png"
            self.image = imageio.imread(self.filepath)
            self.image = cv2.resize(self.image,(50,50))
            self.surpriseScene.clear()
            self.surpriseScene.addPixmap(QPixmap(self.filepath))
            self.surprise.fitInView(self.surpriseScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif(stat == "Neutral"):
            self.filepath = "Color\\Neutral.png"
            self.image = imageio.imread(self.filepath)
            self.image = cv2.resize(self.image,(50,50))
            self.neutralScene.clear()
            self.neutralScene.addPixmap(QPixmap(self.filepath))
            self.neutral.fitInView(self.neutralScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
  
     
    def showGraphClicked(self):
        EmotionRecognition.emotion_analysis(EmotionRecognition.custom)


    def openCameraClicked(self):
        global cam, img, flagCam
        flagCam = 0
        cam = cv2.VideoCapture(0)
        self.openCamera.setDisabled(True)
        if(cam.isOpened() == True):
            self.closeCamera.setDisabled(False)
            self.capturePhoto.setDisabled(False)
        while flagCam==0:
            ret, img = cam.read()
            cv2.imshow('Camera Feed',img)
            k = cv2.waitKey(30)
            if k == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    def liveFeedClicked(self):
        global cam, img, flagCam
        flagCam = 0
        cam = cv2.VideoCapture(0)
        self.openCamera.setDisabled(True)
        if(cam.isOpened() == True):
            self.closeCamera.setDisabled(False)
            self.capturePhoto.setDisabled(True)

        model = models.Sequential()

         #1st convolution layer
        model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
        model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
 
            #2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
            #3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
 
        model.add(Flatten())
 
            #fully connected neural networks
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
 
        model.add(Dense(7, activation='softmax'))

        model.load_weights('facial_expression_model_weights.h5')

        while flagCam==0:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            gray = cv2.resize(gray, (48, 48))

            #img = cv2.resize(img,(48,48))

            img_tensor = image.img_to_array(gray)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            #img_tensor /= 255.
            custom = model.predict(img_tensor)
            
            time.sleep(1)

            maxindex = custom.argmax()
            print(maxindex)

            emotion_array = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            print(emotion_array[maxindex])


            cv2.imshow('Camera Feed',img)
            k = cv2.waitKey(30)
            if k == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

    def closeCameraClicked(self):
        self.openCamera.setDisabled(False)
        self.capturePhoto.setDisabled(True)
        self.closeCamera.setDisabled(True)
        global cam, img, flagCam
        flagCam = 1
        cam.release()
        cv2.destroyAllWindows()

    def capturePhotoClicked(self):
        global cam, img, i
        # Writes image to the location Where i is variable and multiple images can be recorded using by incrementing the variable
        cv2.imwrite(os.path.join('C:/Users/D.CHANDRASHEKHAR/Desktop/jaiiiii/CVIA/Project/EmotionRecognition/EmotionRecognition/CapturedPhotos' , 'CamShot{:d}.jpg').format(i), img)
        i+=1

    def exitClicked(self): 
        window.close()
        QtCore.QCoreApplication.instance().quit()
        sys.exit(app.exec_())
        
if __name__ == "__main__":
        app = QApplication(sys.argv)
        window=EmotionRecognition()
        window.setWindowTitle('Project - Emotion Recognition')
        window.show()
        sys.exit(app.exec_())
