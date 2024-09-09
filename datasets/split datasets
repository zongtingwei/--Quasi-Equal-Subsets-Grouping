import scipy.io as scio
import pandas as pd
import numpy as np

file_path = r'your file path'
data = scio.loadmat(file_path)
dic1 = data['X']
dic2 = data['Y']
df1 = pd.DataFrame(dic1)
df2 = pd.DataFrame(dic2)
df1 = pd.concat([df2, df1], axis=1)
split_ratio = 0.3
split_index = int(len(df1) * split_ratio)
shuffled_df1 = df1.sample(frac=1, random_state=42)
test_data = shuffled_df1[:split_index]
train_data = shuffled_df1[split_index:]
train_data = np.array(train_data)
test_data = np.array(test_data)
save_path1 = r'your save_path'
np.save(save_path1, train_data)
save_path2 = r'your save_path'
np.save(save_path2, test_data)
