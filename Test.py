import os
import cv2
import keras
i=0
x_train, y_train, x_test, y_test = [], [], [], []
filepath="Path for testing Folder"
for i in range(7):
    for filename in os.listdir(os.path.join(filepath,str(i))):
      if filename.endswith(".jpg"):
        if(i==0):
         emotion=0
         print("0thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)
         emotion = keras.utils.to_categorical(emotion, num_classes)
         print(emotion)
         print(img)
        elif(i==1):
         emotion=1
         print("1thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)
        elif(i==2):
         emotion=2
         print("2thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)
        elif(i==3):
         emotion=3
         print("3thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)
        elif(i==4):
         emotion=4
         print("4thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)
        elif(i==5):
         emotion=5
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)
        elif(i==6):
         emotion=6
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)
        elif(i==7):
         emotion=7
         print("7thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_train.append(emotion)
         x_train.append(img)

print("End Of Training Array")

for i in range(7):
    for filename in os.listdir(os.path.join(filepath,str(i))):
      if filename.endswith(".jpg"):
        if(i==0):
         emotion=0
         emotion = keras.utils.to_categorical(emotion, num_classes)
         print("0thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)
        elif(i==1):
         emotion=1
         print("1thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)
        elif(i==2):
         emotion=2
         print("2thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)
        elif(i==3):
         emotion=3
         print("3thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)
        elif(i==4):
         emotion=4
         print("4thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)
        elif(i==5):
         emotion=5
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)
        elif(i==6):
         emotion=6
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)
        elif(i==7):
         emotion=7
         print("7thEmotion")
         path = os.path.join(filepath,str(i))
         img=cv2.imread(os.path.join((os.path.join(filepath,str(i))),filename))
         y_test.append(emotion)
         x_test.append(img)

print("End Of Testing Array")


