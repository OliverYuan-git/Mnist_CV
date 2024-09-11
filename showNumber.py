import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_file_path = "mnist_train.csv"
test_file_path = "mnist_test.csv"
train_data = pd.read_csv(train_file_path)

data_list = train_data.iloc[0].values

# 提取图像数据并将其转换为28x28的二维数组
all_values = data_list
image_array = np.asarray(all_values[1:]).reshape((28, 28))

# 显示图像
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()
