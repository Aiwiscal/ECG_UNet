# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:37:37 2019

@author: Winham

generate_labels.py: 用于生成训练时的标签，将json文件中的信息转换为.npy文件存储
注意：运行前先在同目录下新建一个文件夹119_LABEL

"""

import os
import numpy as np
import json

Mask_path = 'G:/ECG_UNet/119_MASK/'
Label_path = 'G:/ECG_UNet/119_LABEL/'
width = 2378          # 保存图像的宽度（像素数）
sig_length = 1800     # 实际信号长度（采样点数）
N_label_value = 0.5   # 为不同类型定义不同的标记值
V_label_value = 1.0

files = os.listdir(Mask_path)
for i in range(len(files)):
    file_name = files[i]
    print(file_name+' '+str(i+1))
    name = file_name[:-5]
    f = open(Mask_path+file_name, encoding='utf-8')  
    content = json.load(f)['shapes']
    label = np.zeros(sig_length)
    for j in range(len(content)):
        points = content[j]['points']
        # 以下是根据图像宽度和实际信号长度之间的关系计算人工标记的在信号中的实际位置
        start = int(np.round((points[0][0]+points[-1][0])/2.0 / width * sig_length))
        end = int(np.round((points[1][0]+points[-2][0])/2.0 / width * sig_length))
        if content[j]['label'] == 'N':
            label[start:(end+1)] = N_label_value
        else:
            label[start:(end+1)] = V_label_value
                  
    np.save(Label_path+name+'.npy', label)
