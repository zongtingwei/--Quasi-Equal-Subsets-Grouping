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


"""
两个目标函数为例
输入：种群每个解的两个目标函数值
输入：所有非支配前沿
"""


def fast_non_dominated_sort(values1, values2):
    size = len(values1)  # 种群大小
    s = [[] for _ in range(size)]  # 每个解的被支配集合
    n = [0 for _ in range(size)]  # 每个解的支配数
    rank = [0 for _ in range(size)]  # 每个解的等级
    fronts = [[]]  # 所有非支配前沿

    for p in range(size):  # 遍历所有解
        s[p] = []  # 初始化非支配集合和支配数
        n[p] = 0
        for q in range(size):  # 判断p和q支配情况
            # 如果p支配q，增加q到p的被支配集合
            if values1[p] <= values1[q] and values2[p] <= values2[q] \
                    and ((values1[q] == values1[p]) + (values2[p] == values2[q])) != 2:
                s[p].append(q)
            # 如果q支配p，p的支配数+1
            elif values1[q] <= values1[p] and values2[q] <= values2[p] \
                    and ((values1[q] == values1[p]) + (values2[p] == values2[q])) != 2:
                n[p] += 1
        # n[p]=0的解等级设为0，增加到第一前沿
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    # 依次确定其它层非支配前沿
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in s[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        fronts.append(Q)

    del fronts[-1]
    return fronts

# 示例目标函数
def objective1(xi):
    file_path = r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return 1-mean_accuracy

def objective2(xi):
    return np.sum(xi == 1)  # 假设目标是最大化第二个维度

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
# 示例种群
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
population = init_PSO(N, dim)
population_list = population.tolist()
values1 = []
values2 = []
for j in range(N):
    values1.append(objective1(population_list[j]))
    values2.append(objective2(population_list[j]))
# 执行非支配排序
fronts = fast_non_dominated_sort(values1, values2)

# 打印结果
print(fronts)
print("非支配前沿：")
for i, front in enumerate(fronts):
    print(f"前沿 {i+1}: ", [population[idx] for idx in front])
population_fronts = []
for front in fronts:
    for i in range(len(front)):
        a = []
        index = i
        a.append((population_list[index]))
    population_fronts.append(a)
print(population_fronts)
