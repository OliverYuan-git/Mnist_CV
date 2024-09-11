import numpy as np
import pandas as pd


train_file_path = "mnist_train.csv"
test_file_path = "mnist_test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 缩放输入数据，将其从0-255的范围转换到0.01-1.0
inputs = (train_data.iloc[:, 1:].values / 255.0 * 0.99) + 0.01
targets = np.zeros((train_data.shape[0], 10)) + 0.01

for i in range(train_data.shape[0]):
    targets[i][int(train_data.iloc[i, 0])] = 0.99

# 参数
input_nodes = 784
hidden_nodes = 128
output_nodes = 10
learning_rate = 0.1

# 矩阵
w_input_hidden = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
w_hidden_output = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))


# sigmod function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(inputs):
    global w_input_hidden, w_hidden_output  # 声明为全局变量
    hidden_inputs = np.dot(w_input_hidden, inputs)
    hidden_outputs = sigmoid(hidden_inputs)

    final_inputs = np.dot(w_hidden_output, hidden_outputs)
    final_outputs = sigmoid(final_inputs)

    return final_outputs

def train(inputs, targets):
    global w_input_hidden, w_hidden_output  # 声明为全局变量
    hidden_inputs = np.dot(w_input_hidden, inputs)
    hidden_outputs = sigmoid(hidden_inputs)

    final_inputs = np.dot(w_hidden_output, hidden_outputs)
    final_outputs = sigmoid(final_inputs)

    output_errors = targets - final_outputs
    hidden_errors = np.dot(w_hidden_output.T, output_errors)

    # WEIGHT UPDATE
    w_hidden_output += learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
    w_input_hidden += learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)


epochs = 5

for e in range(epochs):
    for i in range(inputs.shape[0]):
        train(inputs[i].reshape(-1, 1), targets[i].reshape(-1, 1))


def test(inputs, targets):
    correct_count = 0
    for i in range(inputs.shape[0]):
        outputs = forward(inputs[i].reshape(-1, 1))
        if np.argmax(outputs) == np.argmax(targets[i]):
            correct_count += 1
    accuracy = correct_count / inputs.shape[0]
    print(f"准确率: {accuracy * 100:.2f}%")


# TEST MODEL
test_inputs = (test_data.iloc[:, 1:].values / 255.0 * 0.99) + 0.01
test_targets = np.zeros((test_data.shape[0], 10)) + 0.01

for i in range(test_data.shape[0]):
    test_targets[i][int(test_data.iloc[i, 0])] = 0.99

test(test_inputs, test_targets)
