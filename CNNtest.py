from __future__ import print_function
import os
os.environ['KERAS_BACKEND']='theano'
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
# from InputImage import load_data
import os
from PIL import Image

def load_data( img_h=32, img_w=32, cron=(65,85,445,395)):
    data = np.empty((420,1,img_h,img_w),dtype="float32")
    label = np.empty((420,),dtype="int32")
    imgs = os.listdir("/home/peter/Keras/movingwindows")
    imgs = sorted(imgs, key=lambda x: int(x.split('_')[1]))
    num = len(imgs)
    for i in range(num):
        img = Image.open("/home/peter/Keras/movingwindows/"+imgs[i])
        img = img.crop(cron)
        img = img.resize( (img_h, img_w), Image.BILINEAR )
        img = img.convert('L')
        img = img.point(lambda x: 0 if x<25 else 1, '1')
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int((imgs[i].split('_')[2]).split('.')[0])
    return data, label

X_train, Y_train  = load_data()
X_test = X_train[:100]
Y_test = Y_train[:100]
X_train = X_train[100:]
Y_train = Y_train[100:]

# input image dimensions
img_rows = X_train.shape[2]
img_cols = X_train.shape[3]
X_train = X_train.reshape( X_train.shape[0], img_rows, img_cols, 1 )
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

batch_size = 128
nb_classes = 3
nb_epoch = 300

# number of convolutional filters to use
nb_filters = 18
# size of pooling area for max pooling
pool_size = (3, 3)
# convolution kernel size
kernel_size = (5, 5)

input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255		#Normalize
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(Y_train, nb_classes)

Y_train = np_utils.to_categorical(Y_train, 3)
Y_test = np_utils.to_categorical(Y_test, 3)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
              
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test Acc:', score[1])
