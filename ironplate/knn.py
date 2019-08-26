import data
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(data.X_data, data.y_data)
pred = knn.predict(data.X_data1)
acc_knn = round(knn.score(data.X_data1, data.y_data1)*100, 2)
print(acc_knn)


# 41.67