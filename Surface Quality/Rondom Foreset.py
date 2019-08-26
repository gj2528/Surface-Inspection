import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# 获取数据
# train_df = pd.read_csv("train.csv")
# # 查看数据类型等信息
# train_df.info()
# # 查看数据内容
# print(train_df.head(10))
# print(train_df.describe())

from sklearn.ensemble import RandomForestClassifier  # use RandomForestRegressor for regression problem
from sklearn.model_selection import train_test_split

# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Random Forest object
model = RandomForestClassifier(n_estimators=1000)
# Train the model using the training sets and check score
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test1.csv")
X = np.array((train_data.as_matrix())[0:, 1:])
Y = X[:, 4]
X = X[:, 0:4]
test_X = np.array((test_data.as_matrix())[0:, 1:])
Y_test = test_X[:, 4]
X_test = test_X[:, 0:4]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
Y = Y * 10
Y_test = Y_test * 10

Y = Y.astype('int')
# print("y_train", Y)
Y_test = Y_test.astype('int')

model.fit(X, Y)
print(model.predict(X_test))
# Predict Output
print('随机森林训练集准确率：\n', model.score(X, Y))
print('随机森林验证集准确率：\n', model.score(X_test, Y_test))  # 分数
