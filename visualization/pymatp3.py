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
import matplotlib.font_manager as fm
# fig, axs = plt.subplots(nrows=2, figsize=(5.5, 3.5), constrained_layout=True)
fig = plt.figure()

ax4 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(234)
ax1 = plt.subplot(235)
ax5 = plt.subplot(133)
classes = ['W', 'LS', 'SWS', 'REM', 'SWS', 'REM']

# confusion_matrix = [[3467, 660, 601, 5, 148, 211],
#  [ 256, 3204, 1006,    0,  161,   62],
#  [ 141,  799, 3540,    0,   43,  179],
#  [   0,    0,    0,    0,    0,    0],
#  [  54,  124,   34,    1,  856,  249],
#  [  41,   20,  108,    1,  250, 1137]]
# confusion_matrix = np.asarray(confusion_matrix)
# proportion = []
# for i in confusion_matrix:
#     for j in i:
#         if np.sum(i) != 0:
#             temp = j / (np.sum(i))
#         else:
#             temp = 0
#         proportion.append(temp)
# # print(np.sum(confusion_matrix[0]))
# # print(proportion)
# pshow = []
# for i in proportion:
#     if i !=0:
#         pt = "%.2f%%" % (i * 100)
#     else:
#         pt = "-"
#     pshow.append(pt)
# proportion = np.array(proportion).reshape(6, 6)  # reshape(列的长度，行的长度)
# pshow = np.array(pshow).reshape(6, 6)
# # print(pshow)
# config = {
#     "font.family": 'Times New Roman',  # 设置字体类型
# }
# rcParams.update(config)
# ax1.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵   PuBu  YlOrRd  YlGnBu  PuRd
# # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
# ax1.set_title('confusion_matrix')
# # plt.colorbar()
# tick_marks = np.arange(len(classes))
# ax1.set_xticks(tick_marks, classes)
# ax1.set_yticks(tick_marks, classes)
#
# thresh = confusion_matrix.max() / 2.
# # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# # ij配对，遍历矩阵迭代器
# iters = np.reshape([[[i, j] for j in range(6)] for i in range(6)], (confusion_matrix.size, 2))
# for i, j in iters:
#     if i != 3:
#         color = 'White'
#     else:
#         color = 'Black'
#     if (i == j):
#         ax1.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9, color = color,
#                  weight=5)  # 显示对应的数字
#         ax1.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9, color = color,)
#     else:
#         ax1.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9)  # 显示对应的数字
#         ax1.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9)
#
# ax1.set_ylabel('True label', fontsize=11)
# ax1.set_xlabel('Predict label', fontsize=11)

myfont = fm.FontProperties(fname='Times_New_Roman.ttf')

#word
confusion_matrix = [[3565, 569,  608,   52,  119,  179],
 [ 653, 3159,  679,    8,  164,   26],
 [ 563,  873, 3037,   13,   63,  153],
 [   0,    0,    0,    0,    0,    0],
 [ 128,  102,   42,   36,  833,  177],
 [ 112,   40,  123,   26,  294,  962]]

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
ax1.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵   PuBu  YlOrRd  YlGnBu  PuRd
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
ax1.set_title('word', fontproperties = myfont)
# plt.colorbar()
tick_marks = np.arange(len(classes))
ax1.set_xticks(tick_marks, classes)
ax1.set_yticks(tick_marks, classes)
ax1.set_xticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'],rotation = 30,fontsize = 'small', fontproperties = myfont)
ax1.set_yticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'], fontsize = 'small', fontproperties = myfont)

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
        ax1.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9, color = color,
                 weight=5)  # 显示对应的数字
        ax1.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9, color = color,)
    else:
        ax1.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9)  # 显示对应的数字
        ax1.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9)

ax1.set_ylabel('True label', fontsize=13, fontproperties = myfont)
ax1.set_xlabel('Predict label', fontsize=13, fontproperties = myfont)


#span
confusion_matrix = [[3481,  606,  686,   43,  127,  149],
 [ 565, 3047,  831,   25,  168,   53],
 [ 562,  691, 3196,   39,   25,  189],
 [   0,    0,    0,    0,    0,    0],
 [  93,  136,   43,   55,  783,  208],
 [  82,   39,  156,   63,  192, 1025]]

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
ax2.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵   PuBu  YlOrRd  YlGnBu  PuRd
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
ax2.set_title('span', fontproperties = myfont, fontsize=11)
# plt.colorbar()
tick_marks = np.arange(len(classes))
ax2.set_xticks(tick_marks, classes)
ax2.set_yticks(tick_marks, classes)
ax2.set_xticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'],rotation = 30,fontsize = 'small', fontproperties = myfont)
ax2.set_yticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'], fontsize = 'small', fontproperties = myfont)
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
        ax2.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9, color = color,
                 weight=5)  # 显示对应的数字
        ax2.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9, color = color,)
    else:
        ax2.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9)  # 显示对应的数字
        ax2.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9)

ax2.set_ylabel('True label', fontsize=11, fontproperties = myfont)
ax2.set_xlabel('Predict label', fontsize=11, fontproperties = myfont)



#bpe
confusion_matrix = [[4145,  252,  366,  158,   64,  107],
 [1958, 1976,  532,   98,   99,   26],
 [2121,  450, 1922,   88,   20,  101],
 [   0,    0,    0,    0,    0,    0],
 [ 225,   60,   25,  345,  523,  140],
 [ 249,   19,   64,  447,  153,  625]]


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
ax3.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵   PuBu  YlOrRd  YlGnBu  PuRd
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
ax3.set_title('bpe', fontproperties = myfont, fontsize=11)
# plt.colorbar()
tick_marks = np.arange(len(classes))
ax3.set_xticks(tick_marks, classes)
ax3.set_yticks(tick_marks, classes)
ax3.set_xticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'],rotation = 30,fontsize = 'small', fontproperties = myfont)
ax3.set_yticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'], fontsize = 'small', fontproperties = myfont)
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
        ax3.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9, color = color,
                 weight=5)  # 显示对应的数字
        ax3.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9, color = color,)
    else:
        ax3.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9)  # 显示对应的数字
        ax3.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9)

ax3.set_ylabel('True label', fontsize=11, fontproperties = myfont)
ax3.set_xlabel('Predict label', fontsize=11, fontproperties = myfont)

#[2]
confusion_matrix = [[3790,  473,  514,   55,  133,  127],
 [ 813, 2964,  691,   17,  168,   36],
 [ 872,  891, 2714,   25,   52, 148],
 [   0,    0,    0,    0,    0,    0],
 [ 105,   86,   16,   90,  835,  186],
 [ 120,   20,   70,  130,  270,  947]]


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
ax4.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵   PuBu  YlOrRd  YlGnBu  PuRd
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
ax4.set_title('W$^2$NER', fontproperties = myfont, fontsize=11)
# plt.colorbar()
tick_marks = np.arange(len(classes))
ax4.set_xticks(tick_marks, classes)
ax4.set_yticks(tick_marks, classes)
ax4.set_xticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'],rotation = 30,fontsize = 'small', fontproperties = myfont)
ax4.set_yticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'], fontsize = 'small', fontproperties = myfont)
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
        ax4.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9, color = color,
                 weight=5)  # 显示对应的数字
        ax4.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9, color = color,)
    else:
        ax4.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9)  # 显示对应的数字
        ax4.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9)

ax4.set_ylabel('True label', fontsize=11, fontproperties = myfont)
ax4.set_xlabel('Predict label', fontsize=11, fontproperties = myfont)




# classes = ['W', 'LS', 'SWS', 'REM']
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
ax5.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵   PuBu  YlOrRd  YlGnBu  PuRd
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
ax5.set_title('HRF', fontproperties = myfont, fontsize=11)
# plt.colorbar()
tick_marks = np.arange(len(classes))
ax5.set_xticks(tick_marks, classes)
ax5.set_yticks(tick_marks, classes)
ax5.set_xticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'],rotation = 30,fontsize = 'small', fontproperties = myfont)
ax5.set_yticklabels(['','O-NON', 'O-CON', 'O-PRO', 'ASP-NON', 'ASP-CON', 'ASP-PRO'], fontsize = 'small', fontproperties = myfont)
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
        ax5.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9, color = color,
                 weight=5)  # 显示对应的数字
        ax5.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9, color = color,)
    else:
        ax5.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=9)  # 显示对应的数字
        ax5.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=9)

ax5.set_ylabel('True label', fontsize=11, fontproperties = myfont)
ax5.set_xlabel('Predict label', fontsize=11, fontproperties = myfont)



plt.tight_layout()
plt.show()
