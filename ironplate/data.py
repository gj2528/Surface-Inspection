from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np

dataPath = './data1.csv'
data1Path = './data2.csv'


# 标准化
scale = False
# trainSize = 0.75

data = np.genfromtxt(dataPath, delimiter=',')
data1 = np.genfromtxt(data1Path, delimiter=',')
# np.random.shuffle(data)

x_data = data[:, 1:]
# x_data = data[:, [5, 6, 8]]
y_data = [int(i * 2 - 0.5) for i in data[:, 0]]
x_data1 = data1[:, 1:]
# x_data = data[:, [5, 6, 8]]
y_data1 = [int(i * 2 - 0.5) for i in data1[:, 0]]

poly_reg = preprocessing.PolynomialFeatures(degree=7)

X_data = poly_reg.fit_transform(x_data)
# dl = int(data.shape[0] * trainSize)
X_data1 = poly_reg.fit_transform(x_data1)

if scale:
    X_data = preprocessing.scale(X_data)
    X_data1 = preprocessing.scale(X_data1)