from imutils import paths
from scipy.io import wavfile
import os
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adadelta

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

clasf = 4
learning_rate = 0.001
epochs = 13

pathtrain = "data/training"
pathtest = "data/testing"

datatrain = []
labeltrain = []
datatesting = []

labeltesting = []

soundPaths = list(paths.list_files(pathtrain))
print("len file sound train: {}".format(len(soundPaths)))
lenS = len(soundPaths)
print("Read file training...")

for soundPath in tqdm(soundPaths):
    label = soundPath.split(os.path.sep)[-2]

#rate: slg diem moi s thu dc
    rate, data = wavfile.read("{}".format(soundPath))
    data = np.array(data, dtype=np.float32)
    nfft = int(rate * 0.05) #ao ra cua so 50mS
    hopl = nfft // 2    #truot 0.25mS
    # data *= 1. / 32768
    # cu 50mS xem tan so xhien bn lan, cu 40 lan 1, sd fast fuire transform chuyen du lieu roi rac tu time sang tso (dc bieu dien duoi dang anh)
    feature = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=40, fmin=0, fmax=8000, n_fft=nfft, hop_length=hopl,
                                   power=2.0) # binh phuong data cho de nhan dang
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(feature, x_axis='time')
    # plt.colorbar()
    # plt.title('MFCC')
    # plt.tight_layout()
    # plt.show()
    feature = np.expand_dims(feature, axis=2)

    datatrain.append(feature)
    labeltrain.append(label)

datatrain = np.array(datatrain, dtype="float32")
labeltrain = np.array(labeltrain)

soundPaths = list(paths.list_files(pathtest))
print("\nlen file sound test: {}".format(len(soundPaths)))
print("Read file testing...")
for soundPath in tqdm(soundPaths):
    label = soundPath.split(os.path.sep)[-2]

    rate, data = wavfile.read("{}".format(soundPath))
    data = np.array(data, dtype=np.float32)
    nfft = int(rate * 0.05)
    hopl = nfft // 2
    # data *= 1. / 32768
    feature = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=40, fmin=0, fmax=8000, n_fft=nfft, hop_length=hopl, power=2.0)
    feature = np.expand_dims(feature, axis=2) # tao the 1 chieu nua
    datatesting.append(feature)
    labeltesting.append(label)


datatesting = np.array(datatesting, dtype="float32")
labeltesting = np.array(labeltesting)


lb = LabelBinarizer()
labeltrain = lb.fit_transform(labeltrain)
labeltesting = lb.transform(labeltesting)

print(datatrain.shape)
print(labeltrain.shape)
print(datatesting.shape)
print(labeltesting.shape)

model = Sequential()

model.add(Conv2D(32, (7, 7), input_shape=(40, 161, 1), padding="SAME", activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding="SAME"))
model.add(Dropout(0.25))
model.add(Conv2D(64, (5, 5), padding="SAME", activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding="SAME"))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="SAME", activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding="SAME"))
model.add(Dropout(0.25))
# model.add(Conv2D(128, (1, 1), padding="SAME", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=clasf, activation='softmax'))

print(model.summary())


opt = SGD(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(datatrain, labeltrain, epochs=epochs, validation_data=(datatesting, labeltesting), batch_size=5)

print("Saving model....")
model.save("phungmodel.h5")
print("Model saved!")
# acc ty le giu so diem du doan dung / tong so diem trong tap data
# loss du doan ko dung/tong
fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig.savefig('accuracy.png')
# plt.show()

fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig.savefig('loss.png')
