# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:49:32 2019

@author: Winham

Unet.py: Unet模型定义


"""

from keras.models import Model
from keras.layers import Input, core, Dropout, concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D


def Unet(nClasses, optimizer=None, input_length=1800, nChannels=1):
    inputs = Input((input_length, nChannels))
    conv1 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(32, 32, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv1D(32, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    conv3 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(128, 32, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv1D(128, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up1 = Conv1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv4))
    merge1 = concatenate([up1, conv3], axis=-1)
    conv5 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv5 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    
    up2 = Conv1D(32, 2, activation='relu', padding='same', kernel_initializer = 'he_normal')(UpSampling1D(size=2)(conv5))
    merge2 = concatenate([up2, conv2], axis=-1)
    conv6 = Conv1D(32, 32, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv1D(32, 32, activation='relu', padding='same')(conv6)
    
    up3 = Conv1D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv6))
    merge3 = concatenate([up3, conv1], axis=-1)
    conv7 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    conv7 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    conv8 = Conv1D(nClasses, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = core.Reshape((nClasses, input_length))(conv8)
    conv8 = core.Permute((2, 1))(conv8)

    conv9 = core.Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    print('\nSummarize the model:\n')
    model = Unet(3)
    model.summary()
    print('\nEnd for summary.\n')
