# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:32:56 2019

@author: Winham

data_preproc.py:用于人工标记后的文件整理
注意：由于下列代码中包含了对文件的删除，因此在原始人工标记后的文件中
仅能运行一次。建议运行前先将原始文件备份！！！若遇到错误可重新恢复并
重新执行。运行前先在同目录下新建一个文件夹119_SEG

"""

import os
import numpy as np
import scipy.io as sio

path = 'G:/ECG_UNet/119_MASK/'               # 原始文件目录
seg_path = 'G:/ECG_UNet/119_SEG/'            # 存储信号.npy文件

files = os.listdir(path)

for i in range(len(files)):
    file_name = files[i]
    print(file_name + ' ' + str(i+1))
    if file_name.endswith('.json'):          # 只取已经人工标记好的信号段，即有.json文件配套
        name = file_name[:-5]
        mat_name = name + '.mat'
        sig = sio.loadmat(path+mat_name)['seg_t'].squeeze()
        np.save(seg_path+name+'.npy', sig)
    elif file_name.startswith('ann') or file_name.endswith('.png'):
        os.remove(path+file_name)

rest_files = os.listdir(path)
for i in range(len(rest_files)):
    if rest_files[i].endswith('.mat'):
        os.remove(path+rest_files[i])
