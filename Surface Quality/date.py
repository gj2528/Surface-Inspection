import pandas as pd
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test1.csv")
X = np.array((train_data.as_matrix())[0:, 1:])
Y = X[:, 4]
X = X[:, 0:4]
test_X = np.array((test_data.as_matrix())[0:, 1:])
Y_test = test_X[:, 4]
X_test = test_X[:, 0:4]
Y = Y * 10
Y_test = Y_test * 10
Y = Y.astype('int')
Y_test = Y_test.astype('int')
