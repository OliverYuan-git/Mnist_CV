import pandas as pd
import numpy as np

train_file_path = r"C:\Users\Oliver\Desktop\numR\mnist_train.csv"
test_file_path = r"C:\Users\Oliver\Desktop\numR\mnist_test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 查看数据结构
print(train_data.head())
print(test_data.head())

# 提取标签和特征
train_labels = train_data.iloc[:, 0].values
train_features = train_data.iloc[:, 1:].values
test_labels = test_data.iloc[:, 0].values
test_features = test_data.iloc[:, 1:].values

# 标准化特征
train_features = train_features / 255.0
test_features = test_features / 255.0

print("训练集大小：", train_features.shape)
print("测试集大小：", test_features.shape)

