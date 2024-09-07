import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

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
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn_classifier, X_selected, y, cv=5)
    mean_accuracy = np.mean(scores)
    return mean_accuracy



def calculate_accuracy(xi):
    file_path = r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return mean_accuracy





def fast_non_dominated_sort(values1, values2):
    size = len(values1) 
    s = [[] for _ in range(size)]  
    n = [0 for _ in range(size)] 
    rank = [0 for _ in range(size)]  
    fronts = [[]]  

    for p in range(size):  
        s[p] = [] 
        n[p] = 0
        for q in range(size):  
            if values1[p] <= values1[q] and values2[p] <= values2[q] \
                    and ((values1[q] == values1[p]) + (values2[p] == values2[q])) != 2:
                s[p].append(q)
            elif values1[q] <= values1[p] and values2[q] <= values2[p] \
                    and ((values1[q] == values1[p]) + (values2[p] == values2[q])) != 2:
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
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


def objective1(xi):
    file_path = r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return 1-mean_accuracy

def objective2(xi):
    return np.sum(xi == 1)  

def init_PSO(pN, dim):
    X = np.zeros((pN, dim))  
    for i in range(pN):  
        for j in range(dim):  
            r = np.random.uniform(0,1)
            if r > 0.5:
                X[i][j] = 1
            else:
                X[i][j] = 0
    return X

data = scio.loadmat(r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat')
dic1 = data['X']
dic2 = data['Y']
df1 = pd.DataFrame(dic1)  
df2 = pd.DataFrame(dic2)  
feats = df1  
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
fronts = fast_non_dominated_sort(values1, values2)
print(fronts)
print("undominated front：")
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
