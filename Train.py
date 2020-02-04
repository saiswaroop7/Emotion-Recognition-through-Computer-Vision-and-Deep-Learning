import os, shutil
import keras

train_dir = 'Path'
validation_dir = 'Path'
test_dir = 'Path/Test'

train_angry_dir = 'Path/0'

train_disgust_dir = 'Path/1'

train_fear_dir = 'Path/2'

train_happy_dir = 'Path/3'

train_sad_dir = 'Path/4'

train_surprise_dir = 'Path/5'

train_neutral_dir = 'Path/6'

validation_angry_dir = 'Path0'

validation_disgust_dir = 'PathPath1'

validation_fear_dir = 'Path2'

validation_happy_dir = 'Path3'

validation_sad_dir = 'Path4'

validation_surprise_dir = 'Path5'

validation_neutral_dir = 'Path6'

test_angry_dir = 'PathTest/0'

test_disgust_dir = 'PathTest/1'

test_fear_dir = 'PathTest/2'

test_happy_dir = 'PathTest/3'

test_sad_dir = 'PathTest/4'

test_surprise_dir = 'PathTest/5'

test_neutral_dir = 'PathTest/6'

from keras import layers
from keras.layers import Dropout

model = models.Sequential()

#1st convolution layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))

#2nd convolution layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

#flatten
model.add(layers.Flatten())

model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers

model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#DATA PREPROCESSING

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(48, 48),
        batch_size=40,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(48, 48),
        batch_size=40,
        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('Data batch shape:', data_batch.shape)
    print('Labels batch shape:', labels_batch.shape)
    break


history = model.fit_generator(
      train_generator,
      steps_per_epoch=200,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=200)

model.save('TRAINED_DATA.h5')