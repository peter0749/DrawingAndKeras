from __future__ import print_function   
from PIL import Image
import numpy as np
from numpy import newaxis
import os
from os import listdir
from os.path import isfile, join
import tkinter as tk
from tkinter.messagebox import showinfo, askyesno
import time
import h5py
os.environ['KERAS_BACKEND']='theano'
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


dirPos='data'
padxlim = 256
padylim = 256
image_count=0
w=[]
pattern_in = np.zeros((padylim,padxlim))
newCl=""
Cl="dog"
trainedModel=[]
label_cat=dict()
label_tac=dict()
labnum=[]
batch_size=128
traindimX = 32
traindimY = 32

def Read_IMGS(width=320, height=240, dirPos='data'):
    imgList = [f for f in listdir(dirPos) if isfile(join(dirPos,f))]
    imgnum = len(imgList)
    labels = []
    imgs = np.empty((imgnum, height, width, 1), dtype="float32")
    i=0
    for path in imgList:
        lab = path.split("-")[2].split(".")[0]
        labels.append(lab)
        img = Image.open(dirPos+'/'+path)
        img = img.resize((width,height), Image.BILINEAR)
        img = img.convert('L')
        print(img.size)
        img = np.asarray(img, dtype='float32')
        img/=255
        print(img.shape)
        img = img[:,:,newaxis]
        imgs[i,:,:,:] = img
        i+=1
    #img.show()
    return imgs, labels

def output_res():
    global Cl,newCl,pattern_in,image_count
    Cl = newCl.get()
    print("New Type: %s\n" % (newCl.get()))
    img = Image.fromarray(pattern_in*255)
    if(img.mode!='RGB'):
        img = img.convert('RGB')
    img.save(dirPos+'/test-'+str(image_count)+'-'+str(newCl.get())+'.bmp')
    image_count+=1

def drawdot(event):
    global w, pattern_in
    if(event.y+3 >= padylim or event.x+3 >= padxlim or event.y-3<0 or event.x-3<0):
        return
    for i in range(-2,2):
        for j in range(-2,2):
            pattern_in[event.y+i][event.x+j]=1.0
    print("%d %d\n" % (event.y, event.x))
    x1, y1 = (event.x-1, event.y-1)
    x2, y2 = (event.x+1, event.y+1)
    w.create_oval(x1,y1,x2,y2)

def reset_bt():
    global pattern_in, w
    pattern_in = np.zeros((padylim,padxlim))
    w.delete(tk.ALL)

def send_toCNN():
    global newCl, Cl, trainedModel
    #trainedModel = load_model('new_model.h5')
    mydraw = np.asarray(pattern_in, dtype='uint8')
    mydraw = Image.fromarray(mydraw*255)
    mydraw = mydraw.resize( (traindimX, traindimY), Image.BILINEAR )
    mydraw = mydraw.convert('L')
    mydraw.show()#Comment this to prevent showing image
    mydraw = np.asarray(mydraw, dtype='float32')
    mydraw/=255
    print(mydraw)
    print(mydraw.shape)
    mydraw = mydraw[newaxis,:,:,newaxis]
    print(mydraw.shape)
    lab = trainedModel.predict_classes(x=mydraw,batch_size=batch_size)
    print(lab[0])
    Cl = label_tac[lab[0]]
    g_pattern="It's "+Cl+"?"
    showinfo('Iguess...',g_pattern)
    if( askyesno('Is it correct?','Choose') ):
        showinfo(':)','Very good!')
    else:
        newWindow=tk.Toplevel()
        newWindow.title("Modify")
        tk.Label(newWindow,text="Correct catagory").pack()
        newCl = tk.Entry(newWindow)
        newCl.pack()
        tk.Button(newWindow, text="Submit", command=output_res).pack(fill=tk.X)

def retrain():
    global label_cat, label_tac, labnum, batch_size, trainedModel, image_count
    imgs, labels = Read_IMGS(width=traindimX, height=traindimY, dirPos=dirPos)
    image_count = len(labels)
#Reset the counter to the very end of the index of image files. And 'image_count' points to the position where the next image appends.
    print(labels)
#Lazy mapping
    label_cat.clear()
    label_tac.clear()
#gotta fix it
    i=0
    for x in labels:
        if(x not in label_cat):
            label_cat[x]=i
            label_tac[i]=x
            i+=1

    nb_classes = len(label_cat)
#For Powerful computers
    nb_epoch = 24
    nb_filters = 40
    pool_size = (2,2)
    kernel_size = (3,3)
#end of comment
    input_shape = (traindimY,traindimX,1)
    imgs = imgs.astype('float32')
    labnum = []
    for x in labels:
        if(x in label_cat):
            labnum.append(label_cat[x])

    print(labnum)
    print('Shape: ', imgs.shape)
    print('Samples: ', imgs.shape[0])
    model = Sequential()
    earlyStop = EarlyStopping(monitor='loss',patience=1, verbose=1, mode='auto')
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adagrad',
                  metrics=['accuracy'])
    model.fit(imgs, labnum, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[earlyStop])
    trainedModel = model
    del model
#Destroy the model generated in function.

root = tk.Tk()
root.resizable(width=False, height=False)
root.title("Painting Input")
w = tk.Canvas(root, width=padxlim, height=padylim)
w.pack(expand=tk.YES, fill=tk.BOTH)
w.bind("<B1-Motion>", drawdot)
submit=tk.Button(root,text="Submit",command=send_toCNN)
submit.pack(side=tk.LEFT)
reset=tk.Button(root,text="Reset",command=reset_bt)
reset.pack(side=tk.RIGHT)
cnnTrigger = tk.Button(root,text="Retrain",command=retrain)
cnnTrigger.pack(side=tk.BOTTOM)
tk.mainloop()
