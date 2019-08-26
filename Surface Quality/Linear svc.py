import date
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(date.X, date.Y)
Y_pred = linear_svc.predict(date.X_test)
acc_linear_svc = round(linear_svc.score(date.X_test, date.Y_test)*100, 2)
print("线性svc:", acc_linear_svc)
