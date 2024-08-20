import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("The two strings must have the same length")

    distance = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            distance += 1

    return distance

def load_and_prepare_data(file_path):
    data = scio.loadmat(file_path)
    X = pd.DataFrame(data['X']).values  
    y = pd.DataFrame(data['Y']).values.ravel()  
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def KNN_with_cross_validation(X_scaled, y, xi):
    boolean_array = np.array(xi).astype(bool)  
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
    file_path = r'your filename.mat'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return mean_accuracy

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

def dominates(individual1, individual2, objectives):
    """
    判断个体1是否支配个体2。
    :param individual1: 个体1。
    :param individual2: 个体2。
    :param objectives: 目标函数列表。
    :return: 如果个体1支配个体2，则返回True。
    """
    better_in_one = False
    for obj in objectives:
        if obj(individual1) > obj(individual2):  # 假设是最大化问题
            return False
        elif obj(individual1) < obj(individual2):
            better_in_one = True
    return better_in_one

# 示例目标函数
def objective1(xi):
    file_path = r'your filename.mat'
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
def calculate_crowding_distance(solutions,ln):
    # 用于计算拥挤距离的函数占位符
    Cis_list = []
    solution_list = []

    for i in range(len(solutions)):
        total_distance_i = 0
        for j in range(len(solutions)-1):
            distance = hamming_distance(solutions[i], solutions[j + 1])
            total_distance_i = total_distance_i + distance
            Cis = total_distance_i / (len(solutions)-1)
            Cis_list.append(Cis)
    sorted_list = sorted(Cis_list, reverse=True)
    for n in range(ln):
        index = Cis_list.index(sorted_list[n])
        solution_list.append(solutions[index])
    return solution_list


def select_feature_subsets(W, Qe):
    # W是特征子集的初始列表，W1, W2, ..., Wk
    # Qe是要处理的查询或任务列表

    # 假设W是列表的列表（每个子列表代表一个特征子集）

    for i in range(len(Qe)):
        # 模拟获取给定查询/任务的TS值
        TS = Qe[i]  # 这实际上应该是调用一个函数，该函数评估Qe[i]并返回TS1, TS2, ..., TSt

        # 对TS解进行非支配排序
        population_list = TS
        values1 = []
        values2 = []
        for j in range(len(population_list)):
            values1.append(objective1(population_list[j]))
        values1 = objective1(population_list)
        values2 = objective2(population_list)
        fronts = fast_non_dominated_sort(values1, values2)
        front_min = fronts[0][0]
        solution = population_list[front_min]
        for j in range (W):
            for k in range(W[j]):
                if solution in W[j][k]:
                    rmin = j
        # 通过方程(5)计算ln - 占位符计算
        ln = len(TS) // len(Qe)  # 占位符，替换为实际方程

        if len(TS) > ln:
            # 计算拥挤距离
            Wrmin = calculate_crowding_distance(TS)
            # 将前ln个特征子集放入Wrmin
            W[rmin].extend(Wrmin)
        else:
            # 将所有TS特征子集放入Wrmin
            W[rmin].extend(TS)
    return W,ln
def calculating_Cio(Wm):
    max = 0
    for i in Wm:
        if Wm[i]>max:
            max = Wm[i]
def calculating_Ci(W,N):
    Cio = 0
    k = len(W)
    for m in range(1, k + 1):
        total_solutions_before_m = sum(len(front) for front in W[:m - 1])
        total_solutions_after_m = sum(len(front) for front in W[:m + 1])
        Cio_list = []
        if total_solutions_before_m < N and total_solutions_after_m > N:
            # 应用刺激条件的操作
            t = []
            for front in W[:m - 1]:
                t.extend(front)
            t_flat = [item for sublist in W for item in sublist]
            objective1_list = []
            objective2_list = []
            for j in range(len(t_flat)):
                objective1_list.append(objective1(t_flat[j]))
                objective2_list.append(objective2(t_flat[j]))
            min_objective1 = min(objective1_list)
            min_objective2 = min(objective2_list)
            max_objective1 = max(objective1_list)
            max_objective2 = max(objective2_list)
            for i in range(len(W)):
                if i == 0 or i == (len(W) - 1):
                    Cio = 1
                    Cio_list.append(Cio)
                else:
                    Cio = objective1(t_flat[i + 1]) - objective1(t_flat[i - 1]) / 2 * (
                                max_objective1 - min_objective1) + objective2(t_flat[i - 1]) - objective2(
                        t_flat[i - 1]) / 2 * (max_objective2 - min_objective2)
                    Cio_list.append(Cio)

        Cis_list = []
        for i in range(len(W[m])):
            total_distance_i = 0
            for j in range(len(W[m]) - 1):
                distance = hamming_distance(W[m][i], W[m][j + 1])
                total_distance_i = total_distance_i + distance
                Cis = total_distance_i / (len(W[m]) - 1)
                Cis_list.append(Cis)
        average_cio = sum(Cio_list)/len(Cio_list)
        average_cis = sum(Cis_list)/len(Cis_list)
        Ci_list = []
        for x in range(len(W[m])):
            if Cio_list[x] > average_cio or Cis_list[x] > average_cis:
                Ci = max(Cio_list, Cis_list)
            else:
                Ci = min(Cio_list, Cis_list)
            Ci_list.append(Ci)
        Ci_list_descending = Ci_list.sort(reverse=True)
        Wm_descending = []
        for y in Ci_list_descending:
            index = Ci_list.index(y)
            Wm_descending.append(W[m][index])
        break
    return Wm_descending



# 还差cis的部分



# 示例种群 
data = scio.loadmat(r'your filename.mat')
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
objectives = [objective1, objective2]
def selected_NP_solutions(W,ln,N):
    NP_solutions = []
    Cis_list = []
    for i in range(len(W)):
        Cio = 0
        k = len(W)
        for m in range(1, k + 1):
            total_solutions_before_m = sum(len(front) for front in W[:m - 1])
            total_solutions_after_m = sum(len(front) for front in W[:m + 1])
            Cio_list = []
            if total_solutions_before_m < N and total_solutions_after_m > N:
                # 应用刺激条件的操作
                t = []
                for front in W[:m - 1]:
                    t.extend(front)
                t_flat = [item for sublist in W for item in sublist]
                objective1_list = []
                objective2_list = []
                for j in range(len(t_flat)):
                    objective1_list.append(objective1(t_flat[j]))
                    objective2_list.append(objective2(t_flat[j]))
                min_objective1 = min(objective1_list)
                min_objective2 = min(objective2_list)
                max_objective1 = max(objective1_list)
                max_objective2 = max(objective2_list)
                for i in range(len(W)):
                    if i == 0 or i == (len(W) - 1):
                        Cio = 1
                        Cio_list.append(Cio)
                    else:
                        Cio = objective1(t_flat[i + 1]) - objective1(t_flat[i - 1]) / 2 * (
                                max_objective1 - min_objective1) + objective2(t_flat[i - 1]) - objective2(
                            t_flat[i - 1]) / 2 * (max_objective2 - min_objective2)
                        Cio_list.append(Cio)
                m1 = m

            Cis_list = []
            for i in range(len(W[m])):
                total_distance_i = 0
                for j in range(len(W[m]) - 1):
                    distance = hamming_distance(W[m][i], W[m][j + 1])
                    total_distance_i = total_distance_i + distance
                    Cis = total_distance_i / (len(W[m]) - 1)
                    Cis_list.append(Cis)
            average_cio = sum(Cio_list) / len(Cio_list)
            average_cis = sum(Cis_list) / len(Cis_list)
            Ci_list = []
            for x in range(len(W[m])):
                if Cio_list[x] > average_cio or Cis_list[x] > average_cis:
                    Ci = max(Cio_list, Cis_list)
                else:
                    Ci = min(Cio_list, Cis_list)
                Ci_list.append(Ci)
            Ci_list_descending = Ci_list.sort(reverse=True)
            Wm_descending = []
            for y in Ci_list_descending:
                index = Ci_list.index(y)
                Wm_descending.append(W[m][index])
    selected_wzero = Wm_descending[:ln]
    for j in range(m1):
        nested_list = W[j]
        flattened_list = [item for sublist in nested_list for item in sublist]
        total_solutions_before_m = sum(len(front) for front in W[:j-1])
        total_solutions_after_m = sum(len(front) for front in W[:j + 1])
        total_solution_m = total_solutions_before_m+ln
        if total_solutions_before_m < N and total_solutions_after_m > N:
            NP_solutions.append(selected_wzero)
        else:
            for k in range(len(flattened_list)):
                NP_solutions.append(flattened_list[k])
    W_rest = W[m1:len(W)-1]
    flat_list = [item for sublist in W_rest for item in sublist]
    for i in range(len(N-total_solution_m)):
        NP_solutions.append(flat_list[i])
    return NP_solutions
def environmental_selection(PO, Qe, N):
    values1 = objective1
    fronts = fast_non_dominated_sort(PO, objectives)
    population_fronts = []
    for front in fronts:
        for i in range(len(front)):
            a = []
            index = i
            a.append((population_list[index]))
        population_fronts.append(a)
    W = population_fronts
    P = []
    W,ln =select_feature_subsets(W, Qe)
    Wm_descending = calculating_Ci(W,N)
    # select NP solutions to P from W1-Wk
    P = selected_NP_solutions(W,ln,N)
    return P
