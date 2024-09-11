import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

train_file_path = "mnist_train.csv"
test_file_path = "mnist_test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 缩放输入数据，将其从0-255的范围转换到0.01-1.0
train_features = (train_data.iloc[:, 1:].values / 255.0)
train_labels = np.zeros((train_data.shape[0], 10))
for i in range(train_data.shape[0]):
    train_labels[i][int(train_data.iloc[i, 0])] = 1

test_features = (test_data.iloc[:, 1:].values / 255.0)
test_labels = np.zeros((test_data.shape[0], 10))
for i in range(test_data.shape[0]):
    test_labels[i][int(test_data.iloc[i, 0])] = 1

# 创建改进后的神经网络模型
model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 使用学习率调度器：学习率在性能不再提升时减小
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# 训练模型
model.fit(train_features, train_labels,
          epochs=50,  # 增加训练轮次
          batch_size=64,  # 调整批次大小
          validation_split=0.2,  # 使用更多的数据进行验证
          callbacks=[lr_scheduler])

# 评估模型
test_loss, test_acc = model.evaluate(test_features, test_labels)
print(f"测试集上的准确率: {test_acc * 100:.2f}%")
