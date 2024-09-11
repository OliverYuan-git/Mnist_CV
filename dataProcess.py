import numpy as np
import pandas as pd

train_file_path = "mnist_train.csv"
test_file_path = "mnist_test.csv"
train_data = pd.read_csv(train_file_path)

# 获取数据集中第一行的数据
all_values = train_data.iloc[0].values

# 缩放输入数据，将其从0-255的范围转换到0.01-1.0
scaled_input = (np.asarray(all_values[1:]) / 255.0 * 0.99) + 0.01

print(scaled_input)

# 对整个数据集进行处理
inputs = (train_data.iloc[:, 1:].values / 255.0 * 0.99) + 0.01
targets = np.zeros((train_data.shape[0], 10)) + 0.01

for i in range(train_data.shape[0]):
    targets[i][int(train_data.iloc[i, 0])] = 0.99

print(inputs.shape)  # 应该输出 (60000, 784)
print(targets.shape) # 应该输出 (60000, 10)
