import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def load_and_prepare_data(file_path):
    # 从文件加载数据
    data = scio.loadmat(file_path)
    X = pd.DataFrame(data['X']).values  # 转换为numpy数组
    y = pd.DataFrame(data['Y']).values.ravel()  # 转换为一维数组

    # 使用 MinMaxScaler 进行特征缩放
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def KNN_with_cross_validation(X_scaled, y, xi):
    # 使用xi选择特征
    boolean_array = np.array(xi).astype(bool)  # 将xi转换为NumPy数组再进行类型转换
    X_selected = X_scaled[:, boolean_array]

    # 创建 k-NN 分类器，设置 k 值
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # 进行5倍交叉验证并计算平均精度
    scores = cross_val_score(knn_classifier, X_selected, y, cv=5)
    mean_accuracy = np.mean(scores)

    return mean_accuracy

# 其他函数和代码保持不变

def calculate_accuracy(xi):
    # 调整为您的数据文件路径
    file_path = r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return mean_accuracy
def objective1(xi):
    file_path = r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return 1-mean_accuracy

def objective2(xi):
    return np.sum(xi == 1)  # 假设目标是最大化第二个维度

def construct_Qe(PO, PO_fit, psi):
    # PO: 初始解的集合
    # PO_fit: 解的适应度值，包括特征子集的FR(f2)值和ER(f1)值
    # psi: ψ-quasi等价的阈值
    PO_feature_fit =[] # PO的选择特征数存储列表
    PO_errorrate_fit = [] # PO的分类错误率存储列表
    PO_list = [] # PO的特征子集存储列表
    for i in range(len(PO_fit)): # 对应标签存储
        PO_feature_fit.append(PO_fit[i][1])
        PO_errorrate_fit.append(PO_fit[i][0])
        PO_list.append(PO[i])

    # 2. 根据FR(f2)值将PO划分为m个组
    # 初始化空列表和变量
    groups = []
    groups_features = []
    groups_errorrate = []
    groups_solutions = []
    for i in range(len(PO_feature_fit)):
        if PO_feature_fit[i] not in groups_features:
            groups_features.append(PO_feature_fit[i])
            a=[]
            a.append(PO_errorrate_fit[i])
            b=[]
            b.append(PO_list[i])
            groups_errorrate.append(a)
            groups_solutions.append(b)
        else:
            index = groups_features.index(PO_feature_fit[i])
            groups_errorrate[index].append(PO_errorrate_fit[i])
            groups_solutions[index].append(PO_list[i])
        print(groups_features)
        print(groups_errorrate)
        print(groups_solutions)
    for i in range(len(groups_features)):
        a=[]
        a.append(groups_solutions[i])
        a.append(groups_errorrate[i])
        a.append(groups_features[i])
        groups.append(a)
        print(groups)
    m = len(groups_features)

    # 3. 构建一个空的集合Qe，长度为m
    Qe = [[] for _ in range(m)]

    for i, group in enumerate(groups):
        ai = [error_rate for error_rate in group[1]]
        Smin = min(ai)
        # 检查当前组中的其他子集是否与Smin是ψ-quasi等价的
        for k, error_rate in enumerate(ai):
            if abs(error_rate - Smin) <= psi:
                Qe[i].append(group[0][k])
            print(Qe)

        # 检查当前组中的其他子集是否与Smin是ψ-quasi等价的

    # 10. 移除Qe中的空子集
    Qe = [subset for subset in Qe if subset]
    return Qe
def init_PSO(pN, dim):
    X = np.zeros((pN, dim))  # 所有粒子的位置
    for i in range(pN):  # 外层循环遍历种群中每个粒子
        for j in range(dim):  # 内层循环遍历种群中粒子的每个维度
            r = np.random.uniform(0,1)
            if r > 0.5:
                X[i][j] = 1
            else:
                X[i][j] = 0
    return X

data = scio.loadmat(r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat')
dic1 = data['X']
dic2 = data['Y']
df1 = pd.DataFrame(dic1)  # 将NumPy数组转换为DataFrame对象
df2 = pd.DataFrame(dic2)  # 将NumPy数组转换为DataFrame对象
feats = df1  # 导入特征数据集
labels = df2
dim = len(df1.columns)
MAX_FE = 250
N = 20
PO = init_PSO(N, dim)
PO_fit = []
for i in range(N):
    PO_i = []
    PO_i.append(objective1(PO[i]))
    PO_i.append(objective2(PO[i]))
    PO_fit.append(PO_i)
values1 = []
values2 = []
for j in range(N):
    values1.append(objective1(PO[j]))
    values2.append(objective2(PO[j]))
Qe = construct_Qe(PO, PO_fit, 0.1)
print(pd.DataFrame(Qe))
print(values1)
print(values2)
