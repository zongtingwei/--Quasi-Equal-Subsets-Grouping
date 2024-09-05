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
    
def objective1(xi):
    file_path = r'C:\Users\11741\Pycharm\pythonProject1\data\colon.mat'
    X_scaled, y = load_and_prepare_data(file_path)
    mean_accuracy = KNN_with_cross_validation(X_scaled, y, xi)
    return 1-mean_accuracy

def objective2(xi):
    return np.sum(xi == 1)  

def construct_Qe(PO, PO_fit, psi):
    PO_feature_fit =[] 
    PO_errorrate_fit = [] 
    PO_list = []
    for i in range(len(PO_fit)): 
        PO_feature_fit.append(PO_fit[i][1])
        PO_errorrate_fit.append(PO_fit[i][0])
        PO_list.append(PO[i])
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
    Qe = [[] for _ in range(m)]
    for i, group in enumerate(groups):
        ai = [error_rate for error_rate in group[1]]
        Smin = min(ai)
        for k, error_rate in enumerate(ai):
            if abs(error_rate - Smin) <= psi:
                Qe[i].append(group[0][k])
            print(Qe)
    Qe = [subset for subset in Qe if subset]
    return Qe
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
