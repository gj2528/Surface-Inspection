import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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
# model = KNeighborsClassifier()
# K_value = 2
# neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
# neigh.fit(X, Y)
# testt = neigh.predict(X_test)
# print(testt)
# scoree = accuracy_score(Y_test, testt)
# print("KNN testing accuracy=", scoree * 100)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X, Y)
pred = knn.predict(X_test)
acc_knn = round(knn.score(X_test, Y_test)*100, 2)
print(acc_knn)


# 32.06