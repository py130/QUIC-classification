# --coding:utf-8--

import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

stored_model_path = "stored_model/xgb.pkl"

def xgb_model(x, y, val_x, val_y):
    dtrain = xgb.DMatrix(x, y)
    dval = xgb.DMatrix(val_x, val_y)
    dtest = xgb.DMatrix(val_x)

    num_rounds = 2000  # 迭代次数
    params = {'booster': 'gbtree',
              'eta': 0.8,
              'max_depth': 5,
              'objective': 'multi:softmax',  # 'binary:logistic',
              'eval_metric': 'merror',
              'num_class': 9,
              #  'min_child_weight':2,
              #  'gamma':0.2,
              #  'subsample':0.9,
              #  'colsample_bytree':0.9,
              #  'nthread':4,
              #  'scale_pos_weight':1,
              'random_seed': 2018
              }
    # 训练集和测试集
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    # 模型训练
    xgb_model = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=50)
    # 模型验证
    pre_y = xgb_model.predict(dtest, ntree_limit=xgb_model.best_ntree_limit)
    print(confusion_matrix(val_y, pre_y))
    # 计算准确率:
    print("xgb model accuracy:", metrics.accuracy_score(val_y, pre_y))
    print(classification_report(val_y, pre_y))
    # 保存模型
    pickle.dump(xgb_model, open(stored_model_path, "wb"))


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

xgb_model(x, y, val_x, val_y, )
