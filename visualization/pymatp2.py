# -*- coding: utf-8 -*-
"""
@Time    : 2021/11/18 0:33
@Author  : ONER
@FileName: plt_cm.py
@SoftWare: PyCharm
"""

# confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5),
                        constrained_layout=True)

classes = ['W', 'LS', 'SWS', 'REM']
confusion_matrix = np.array([(193, 31, 0, 41), (87, 1038, 32, 126), (17, 337, 862, 1), (17, 70, 0, 638)],
                            dtype=np.int)  # 输入特征矩阵
confusion_matrix = [[3467, 660, 601, 5, 148, 211],
 [ 256, 3204, 1006,    0,  161,   62],
 [ 141,  799, 3540,    0,   43,  179],
 [   0,    0,    0,    0,    0,    0],
 [  54,  124,   34,    1,  856,  249],
 [  41,   20,  108,    1,  250, 1137]]
confusion_matrix = np.asarray(confusion_matrix)
proportion = []
for i in confusion_matrix:
    for j in i:
        if np.sum(i) != 0:
            temp = j / (np.sum(i))
        else:
            temp = 0
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
# print(proportion)
pshow = []
for i in proportion:
    if i !=0:
        pt = "%.2f%%" % (i * 100)
    else:
        pt = "-"
    pshow.append(pt)
proportion = np.array(proportion).reshape(6, 6)  # reshape(列的长度，行的长度)
pshow = np.array(pshow).reshape(6, 6)
# print(pshow)
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵   PuBu  YlOrRd  YlGnBu  PuRd
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

thresh = confusion_matrix.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# ij配对，遍历矩阵迭代器
iters = np.reshape([[[i, j] for j in range(6)] for i in range(6)], (confusion_matrix.size, 2))
for i, j in iters:
    if i != 3:
        color = 'White'
    else:
        color = 'Black'
    if (i == j):
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12, color = color,
                 weight=5)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color = color,)
    else:
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12)  # 显示对应的数字
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12)

plt.ylabel('True label', fontsize=16)
plt.xlabel('Predict label', fontsize=16)
plt.tight_layout()
plt.show()
