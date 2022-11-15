# QUIC协议识别


运行流程：

1. 使用示例数据集流程：
    1. 运行 cnn1d.py/xbg.py/rf.py 三个程序，训练模型、保存到 stored_model 文件夹
   

2. 使用完整数据集流程：
    1. 到 [QUIC Dataset](https://drive.google.com/drive/folders/1Pvev0hJ82usPh6dWDlz7Lv8L6h3JpWhE) 下载数据集，将其中的 pretraining 文件夹的文件复制到本项目的 data-all 文件夹（可省略）
    2. processData.py 内的参数配置(可省略)：
       1. dirPath="data-all"
       2. xdatanpyPath="datanpy-all/xdata.npy"
       3. ydatanpyPath="datanpy-all/ydata.npy"
    3. 运行processData.py（可省略）
    3. cnn1d.py/xbg.py/rf.py 内的参数配置（必须）：
       1. xdataPath="datanpy-all/xdata.npy"
       2. ydataPath="datanpy-all/ydata.npy"
    3. 运行 cnn1d.py/xbg.py/rf.py 三个程序（必须）


[参考论文](https://github.com/shrezaei/Semi-supervised-Learning-QUIC-)