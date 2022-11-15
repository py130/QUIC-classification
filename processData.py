# --coding:utf-8--
# 将QUIC数据集的数据，处理成npy数组，用于模型输入

import pdb
import os
import numpy as np

# 示例的
dirPath = "data/"

labelDict = {"Google Doc": 0,
             "Google Drive": 1,
             "Google Music": 2,
             "Google Search": 3,
             "Youtube": 4
             }
# 每个会话取的前面的若干个包的个数，默认为50
FIXLEN = 50
# 数据
xdata = np.array([])
# 标签
ydata = np.array([])

for folder in os.listdir(dirPath):
    FileCounter = 0
    print(folder)
    label = labelDict.get(folder)
    foldername = dirPath + folder
    for file in os.listdir(foldername):
        filename = foldername + "/" + file
        with open(filename) as f:
            FileCounter += 1
            statFeatures = np.zeros(42)
            EntireFile = []
            for line in f:
                data = line.split()
                try:
                    EntireFile.append(data)
                except:
                    print(EntireFile)
                    pdb.set_trace()
            try:
                EntireFile = np.array(EntireFile).astype(np.float)
            except:
                print(EntireFile)
                pdb.set_trace()
            # 每个会话取前FIXLEN个包，过长则截断
            result = EntireFile[:FIXLEN, 1:3]
            # 不足L则补0
            if result.__len__() < FIXLEN:
                result = np.concatenate([result, np.zeros([FIXLEN - result.__len__(), 2])], axis=0)
            # result = result.reshape([1, 100])
            xdata = np.append(xdata, result)
            ydata = np.append(ydata, label)

# 按照模型需要的格式（会话数, 1, 特征）将每个会话分开,默认每个会话取前50个包的相对时间戳和包长，所以共100个特征。
xdata = xdata.reshape(-1, 1, FIXLEN * 2)

# -------------configure----------------
xdatanpyPath = "datanpy/xdata.npy"
ydatanpyPath = "datanpy/ydata.npy"
# -------------configure----------------

np.save(xdatanpyPath, xdata)
np.save(ydatanpyPath, ydata)
