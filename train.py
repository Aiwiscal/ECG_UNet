# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:52:07 2019

@author: Administrator

train.py: 训练模型

"""

from Unet import Unet
import LoadBatches1D
import keras
from keras import optimizers
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def lr_schedule(epoch):
    # 训练网络时学习率衰减方案
    lr = 0.0001
    if epoch >= 50:
        lr = 0.00001
    print('Learning rate: ', lr)
    return lr


train_sigs_path = 'G:/ECG_UNet/train_sigs/'
train_segs_path = 'G:/ECG_UNet/train_labels/'
train_batch_size = 1
n_classes = 3
input_length = 1800
optimizer_name = optimizers.Adam(lr_schedule(0))
val_sigs_path = 'G:/ECG_UNet/val_sigs/'
val_segs_path = 'G:/ECG_UNet/val_labels/'
val_batch_size = 2

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

model = Unet(n_classes, input_length=input_length)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_name,
              metrics=['accuracy'])

model.summary()

output_length = 1800

G = LoadBatches1D.SigSegmentationGenerator(train_sigs_path, train_segs_path, train_batch_size, n_classes, output_length)

G2 = LoadBatches1D.SigSegmentationGenerator(val_sigs_path, val_segs_path, val_batch_size, n_classes, output_length)

checkpointer = keras.callbacks.ModelCheckpoint(filepath='myNet.h5', monitor='val_acc', mode='max', save_best_only=True)

history = model.fit_generator(G, 500//train_batch_size, validation_data=G2, validation_steps=200, epochs=70,
                        callbacks=[checkpointer, lr_scheduler])

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
