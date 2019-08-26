import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

import data

from sklearn.ensemble import RandomForestClassifier  # use RandomForestRegressor for regression problem
from sklearn.model_selection import train_test_split


model = RandomForestClassifier(n_estimators=1000)


model.fit(data.X_data, data.y_data)
print(model.predict(data.X_data1))
# Predict Output
print('随机森林训练集准确率：\n', model.score(data.X_data, data.y_data))
print('随机森林验证集准确率：\n', model.score(data.X_data1, data.y_data1))  # 分数
