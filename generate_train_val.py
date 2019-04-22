# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:40:34 2019

@author: Winham

generate_train_val.py: 生成训练集和验证集
注意：运行前新建文件夹train_sigs,train_labels,val_sigs,val_labels


"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

Sig_path = 'G:/ECG_UNet/119_SEG/'
Label_path = 'G:/ECG_UNet/119_LABEL/'

train_sig_path = 'G:/ECG_UNet/train_sigs/'
train_label_path = 'G:/ECG_UNet/train_labels/'
val_sig_path = 'G:/ECG_UNet/val_sigs/'
val_label_path = 'G:/ECG_UNet/val_labels/'

sig_files = os.listdir(Sig_path)
label_files = os.listdir(Label_path)

sig_files.sort()
label_files.sort()

sig_train, sig_val, label_train, label_val = train_test_split(
        sig_files, label_files, test_size=100, random_state=42)  # 训练集500，验证集100

for i in range(len(sig_train)):
    print('Train No.'+str(i+1)+':'+sig_train[i])
    sig = np.load(Sig_path+sig_train[i])
    label = np.load(Label_path+label_train[i])
    np.save(train_sig_path+sig_train[i], sig)
    np.save(train_label_path+label_train[i], label)

for i in range(len(sig_val)):
    print('Val No.'+str(i+1)+':'+sig_val[i])
    sig = np.load(Sig_path+sig_val[i])
    label = np.load(Label_path+label_val[i])
    np.save(val_sig_path+sig_val[i], sig)
    np.save(val_label_path+label_val[i], label)
