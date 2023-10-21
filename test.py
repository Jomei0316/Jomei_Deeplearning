# 导入数据集
# 处理数据集维度
# 处理标签
# 对数据集标准化
# 定义sigmoid函数
# 初始化w b
# 定义前向传播、其中计算A和损失函数，计算梯度
# 定义优化函数  通过梯度下降更新wb 返回损失函数的历史记录
# 定义预测函数，对数据进行预测
# 整合训练集和测试过程

import h5py
import matplotlib.pyplot as plt
import numpy as np

train_data = h5py.File('D:/BaiduNetdiskDownload/the frist week/train_catvnoncat.h5', 'r') #r代表只读
test_data = h5py.File('D:/BaiduNetdiskDownload/the frist week/test_catvnoncat.h5', 'r')

# print(train_data['train_set_x'])

#取数据
train_data_org = train_data['train_set_x'][:]
train_label_org = train_data['train_set_y'][:]
test_data_org = test_data['test_set_x'][:]
test_label_org = test_data['test_set_y'][:]

#处理数据维度 转换为12288*1的格式
m_train = train_data_org.shape[0]
m_test = test_data_org.shape[0]
train_data_trans = train_data_org.reshape(m_train, -1).T #-1的作用是自动计算出矩阵的维度，因此得到了一个矩阵维度为m_train, 特征的数字，即12288 * 209
# print(train_data_trans.shape[1]) #shape 0是列数 shape 1是行数
test_data_trans = test_data_org.reshape(m_test, -1).T
# print(train_data_trans.shape)
# print(test_data_trans.shape)

#训练标签
train_label_trans = train_label_org[np.newaxis, :]
test_label_trans = test_label_org[np.newaxis, :]





#进行数据标准化
train_data_sta = train_data_trans / 255
test_data_sta = test_data_trans / 255

#定义sigmoid函数
def sigmoid(z):
    fun = 1 / (1 + np.exp(-z))
    return fun

#初始化参数w, b
n_dim = train_data_sta.shape[0]
w = np.zeros((n_dim, 1))
b = 0

#定义前向传播函数、损失函数、梯度下降
def propagate(w, b, X, Y):
    #1.定义前向传播函数
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    #2.计算代价函数
    m = X.shape[1]
    J = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    #3.梯度下降
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    #设计一个字典存储dw和db
    grad = {'dw' : dw, 'db' : db}

    return grad, J

#定义优化函数
def optimizer(w, b, X, Y, alpha, n_iters): #n_iters指的是迭代次数
    cost = []
    for i in range(n_iters):
        grad, J = propagate(w, b, X, Y)
        dw = grad['dw']
        db = grad['db']

        w = w - alpha * dw
        b = b - alpha * db

        #每隔100次打印一下cost
        if  i % 100 == 0:
            cost.append(J)
            print('训练次数为：', i, '损失为：', J)

    #返回梯度
    grad = {'dw': dw, 'db' : db}
    params = {'w' : w, 'b' : b}
    return grad, params, cost

#预测部分，利用刚才定义好的w, b来对图像预测
def predict(w, b, X_test):
    z = np.dot(w.T, X_test) + b
    A = sigmoid(z)

    m = X_test.shape[1]
    y_pred = np.zeros((1, m))
    for i in range(m):
        if A[:, 1] > 0.5:
            y_pred[:, i] = 1
        else:
            y_pred[:, i] = 0
    return y_pred

#整合模型
def model(w, b, X_train, Y_train, X_test, Y_test, alpha, n_iters):
    grad, params, cost = optimizer(w, b, X_train, Y_train, alpha, n_iters)

    w = params['w']
    b = params['b']
    y_pred_train = predict(w, b, X_train)
    y_pred_test = predict(w, b, X_test)

    #查看训练集和测试集的准确率
    print("train acc:", np.mean(y_pred_train == Y_train) * 100, '%')
    print("train acc:", np.mean(y_pred_test == Y_test) * 100, '%')

    b = {
        'w' : w,
        'b' : b,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'alpha' : alpha
    }
    return b

b = model(w, b, train_data_sta, train_label_org, test_data_sta, test_label_trans, 0.1, 2000)






























