import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
# matplotlib inline
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

train_data = '/home/sebastien/Euros_Recognition/train'
test_data = '/home/sebastien/Euros_Recognition/tests'


def one_hot_label(img):
    label = img.split('.')[0]
    if label == '1cent':
        ohl = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    elif label == '2cent':
        ohl = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    elif label == '5cent':
        ohl = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    elif label == '10cent':
        ohl = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    elif label == '20cent':
        ohl = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    elif label == '50cent':
        ohl = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    elif label == '1eur':
        ohl = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    elif label == '2eur':
        ohl = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    return ohl


def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images


def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        test_images.append([np.array(img), one_hot_label(i)])
    return test_images


training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 3)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 3)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = Sequential()

# model.add(InputLayer(input_shape=[64,64,1]))
model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(
    Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(64, 64, 3)))  # 50
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(
    Conv2D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(64, 64, 3)))  # 80
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(8, activation='softmax'))
optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=50, batch_size=100)
model.summary()

fig = plt.figure(figsize=(14, 14))
count = 0

for cnt, data in enumerate(testing_images[10:40]):
    y = fig.add_subplot(6, 5, cnt + 1)
    img = data[0]
    data = img.reshape(1, 64, 64, 3)
    model_out = model.predict([data])
    if np.argmax(model_out) == 0:
        str_label = '1cent'
        count += 1
    elif np.argmax(model_out) == 1:
        str_label = '2cent'
        count += 2
    elif np.argmax(model_out) == 2:
        str_label = '5cent'
        count += 5
    elif np.argmax(model_out) == 3:
        str_label = '10cent'
        count += 10
    elif np.argmax(model_out) == 4:
        str_label = '20cent'
        count += 20
    elif np.argmax(model_out) == 5:
        str_label = '50cent'
        count += 50
    elif np.argmax(model_out) == 6:
        str_label = '1eur'
        count += 100
    elif np.argmax(model_out) == 7:
        str_label = '2eur'
        count += 200
    y.imshow(img, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
count = count / 100
print('Il y a ', count, ' euros dans ce porte monnaie')

# serialize model to JSON
model_json = model.to_json()
with open("model_test.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_test.h5")
print("Saved model to disk")
