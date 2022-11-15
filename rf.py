# --coding:utf-8--

import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier

stored_model_path = "stored_model/rf.pkl"

def rf_model(x, y, val_x, val_y):
    # 默认参数：
    model = RandomForestClassifier(random_state=2018)
    # 训练模型:
    model.fit(x, y)
    # 模型验证：
    pre_y = model.predict(val_x)
    print(confusion_matrix(val_y, pre_y))
    # 计算准确率:
    print("RandomForestClassifier model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))
    # 保存模型
    pickle.dump(model, open(stored_model_path, "wb"))


def change_type(data):
    xx = []
    for i in range(len(data)):
        x = []

        for k in range(len(data[i])):
            x.extend(data[i][k])
        xx.append(x)
    return xx

# -------------configure----------------
xdataPath = "datanpy-all/xdata.npy"
ydataPath = "datanpy-all/ydata.npy"
# -------------configure----------------

xdata = np.load(xdataPath)
ydata = np.load(ydataPath)
train_x = change_type(xdata)
train_y = ydata.tolist()
x, val_x, y, val_y = train_test_split(train_x, train_y, test_size=0.4, random_state=2018, stratify=train_y)

rf_model(x, y, val_x, val_y)
