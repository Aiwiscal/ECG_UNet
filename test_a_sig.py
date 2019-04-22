# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:08:55 2019

@author: Winham

test_a_sig.py: 加载训练好的模型，从验证集中随机选取一条信号进行测试

"""

import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
from sklearn import preprocessing as prep
import matplotlib.pyplot as plt
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

val_sig_path = 'G:/ECG_UNet/val_sigs/'
val_label_path = 'G:/ECG_UNet/val_labels/'

sig_files = os.listdir(val_sig_path)
label_files = os.listdir(val_label_path)

select = np.random.choice(sig_files, 1)[0]

a_sig = np.load(val_sig_path+select)
a_seg = np.load(val_label_path+select)

K.clear_session()
tf.reset_default_graph()
model = load_model('myNet.h5')

a_sig = np.expand_dims(prep.scale(a_sig), axis=1)
a_sig = np.expand_dims(a_sig, axis=0)

tic = time.time()
a_pred = model.predict(a_sig)
toc = time.time()


print('Elapsed time: '+str(toc-tic)+' seconds.')
plt.plot(a_sig[0, :, 0])
plt.grid(True)

plt.plot(a_pred[0, :, 0], 'b')
plt.plot(a_pred[0, :, 1], 'k')
plt.plot(a_pred[0, :, 2], 'r')
plt.legend(['Sig', 'Background', 'Normal', 'PVC'], loc='lower right')
