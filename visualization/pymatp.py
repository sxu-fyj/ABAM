from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_pred = ['2','2','3','1','4'] # ['2','2','3','1','4'] # 类似的格式
y_true = ['0','1','2','3','4'] # ['0','1','2','3','4'] # 类似的格式
# 对上面进行赋值

C = confusion_matrix(y_true, y_pred, labels=['0','1','2','3','4']) # 可将'1'等替换成自己的类别，如'cat'。
cm = [[3467, 660, 601, 5, 148, 211],
 [ 256, 3204, 1006,    0,  161,   62],
 [ 141,  799, 3540,    0,   43,  179],
 [   0,    0,    0,    0,    0,    0],
 [  54,  124,   34,    1,  856,  249],
 [  41,   20,  108,    1,  250, 1137]]

plt.matshow(cm, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
# plt.colorbar()

for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.annotate(cm[j][i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
plt.yticks(range(0,5), labels=['a','b','c','d','e'])
plt.show()
