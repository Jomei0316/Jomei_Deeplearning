#导入数据
# 导入必要的库，包括 h5py 用于处理HDF5格式的数据文件，matplotlib 用于可视化，以及 numpy 用于数学运算。
#
# 从HDF5格式的数据文件中导入训练数据集和测试数据集。
#
# 处理数据的维度，将每个样本的特征转换为列向量，以便后续的矩阵运算。
#
# 处理标签，将训练标签和测试标签变成形状为 (1, 样本数) 的数组。
#
# 对数据进行标准化，将像素值缩放到0到1的范围内。
#
# 定义sigmoid函数，它将在模型的前向传播中使用。
#
# 初始化模型参数 w 和 b。
#
# 定义前向传播函数 propagate，其中计算预测 A 和损失函数 J，并计算梯度 dw 和 db。
#
# 定义优化函数 optimize，它使用梯度下降来更新模型参数 w 和 b，并返回损失函数的历史记录。
#
# 定义预测函数 predict，它根据学习到的参数对新的数据进行预测。
#
# 整合模型的训练和测试过程，包括调用 optimize 进行参数更新，计算训练和测试的准确率，然后返回模型参数和其他相关信息。
#
# 这个代码片段涵盖了逻辑回归模型的关键部分，包括数据的处理、模型参数的初始化和更新、前向传播、损失计算以及训练和测试的过程。你可以使用这个模型来进行二元分类任务，例如图像中是否包含猫。
import h5py
import matplotlib.pyplot as plt
import numpy as np

eplilon = 1e-5

#训练原始文件
train_dataset = h5py.File('D:/BaiduNetdiskDownload/the frist week/train_catvnoncat.h5', 'r')
#测试原始文件
test_dataset = h5py.File('D:/BaiduNetdiskDownload/the frist week/test_catvnoncat.h5', 'r')


for key in train_dataset.keys():
    print(key)
#一般x表示输入特征，y表示标签            209张训练图片 64*64*3
print(train_dataset['train_set_x'].shape)
print(train_dataset['train_set_y'].shape) #y就是第几张图

for key in test_dataset.keys():
    print(key)
#50张测试图片
print(test_dataset['test_set_x'].shape)
print(test_dataset['test_set_y'].shape)

#取出训练集和测试集
train_dataset_org = train_dataset['train_set_x'][:] #取出原图
train_labels_org = train_dataset['train_set_y'][:]
test_dataset_org = test_dataset['test_set_x'][:]
test_labels_org = test_dataset['test_set_y'][:]

#查看图片
# %matplotlib inline #在线显示图片
# plt.imshow(train_dataset_org[176])
# plt.show()

#数据维度的处理
m_train = train_dataset_org.shape[0]    #获取训练数据集的行数
m_test = test_dataset_org.shape[0]      #获取测试数据集的行数
print(m_train, m_test)
train_data_trans = train_dataset_org.reshape(m_train, -1).T #转成这样的格式 209不变，后面变成列向量  吴恩达讲过
test_data_trans = test_dataset_org.reshape(m_test, -1).T
print(train_data_trans.shape) #12288个特征，209个样本
print(test_data_trans.shape)

#处理标签
#训练标签
train_labels_trans = train_labels_org[np.newaxis, :]# 1行50列
test_labels_trans = test_labels_org[np.newaxis, :]

#数据标准化
print(test_data_trans)
train_data_sta = train_data_trans / 255
test_data_sta = test_data_trans / 255
print(train_data_sta.shape)

# 总结：
# 导入包 -> 导入数据集 -> 取出数据集（包括原始数据和label) -> 处理数据维度（转换成这种形式：每个列向量都是每张图片的特征） -> 处理标签（转换成1行50列） -> 数据标准化

#定义sigmoid函数
def sigmoid(z):
    a = 1 / (1 + np.exp(-z)) #np.exp代表指数函数
    return a

#初始化参数w， b
n_dim = train_data_sta.shape[0]
# print(n_dim)
w = np.zeros((n_dim, 1)) #定义为n行1列的0矩阵
b = 0

#定义前向传播函数、代价函数、梯度下降
def propagate(w, b, X, y):

    #1.定义前向传播函数
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    #2.计算代价函数
    #获取样本个数
    m = X.shape[1]
    J = -1/m * np.sum(y * np.log(A+eplilon) + (1-y) * np.log(1-A+eplilon))

    #3.梯度下降
    dw = 1 / m * np.dot(X, (A - y).T)
    db = 1 / m * np.sum(A - y)
    grads = {'dw' : dw, 'db' : db}

    return grads, J

#优化部分
def optimize(w, b, X, y, alpha, n_iters):
    costs = []
    for i in range(n_iters):
        grads, J = propagate(w, b, X, y)
        dw = grads['dw']
        db = grads['db']

        w = w - alpha * dw
        b = b - alpha * db

        #每隔xx次打印一次代价函数
        if i % 100 == 0:
            costs.append(J)
            print('n_iters is', i, 'cost is', J)
    #返回梯度
    grads = {'dw': dw, 'db': db}
    params = {'w': w, 'b': b}
    return grads, params, costs

#预测部分  使用刚才定义好的w， b来对新的测试集进行预测

def predict(w, b, X_test):
    z = np.dot(w.T, X_test) + b
    A = sigmoid(z)
    #取样本数
    m = X_test.shape[1]
    y_pred = np.zeros((1, m))

    for i in range (m):
        if A[:, i] > 0.5:
            y_pred[:, i] = 1
        else:
            y_pred[:, i] = 0
    return y_pred

#整合模型
def model(w, b, X_train, y_train, X_test, y_test, alpha, n_iters): #训练集数据，训练集标签，测试集同理， 学习速率， 迭代次数
    grads, params, costs = optimize(w, b, X_train, y_train, alpha, n_iters)

    w = params['w']
    b = params['b']
    #在训练集和测试集上进行预测
    y_pred_train = predict(w, b, X_train)
    y_pred_test = predict(w, b, X_test)
    #查看训练集和测试集的准确率
    print('train acc:', np.mean(y_pred_train == y_train) * 100, "%")
    print('test acc: ', np.mean(y_pred_test == y_test) * 100, '%')

    b = {
        'w' : w,
        'b' : b,
        'y_pred_train' : y_pred_train,
        'y_pred_test' : y_pred_test,
        'alpha' : alpha

    }
    return b

b = model(w, b, train_data_sta, train_labels_trans, test_data_sta, test_labels_trans, alpha=0.5, n_iters=2000)


















