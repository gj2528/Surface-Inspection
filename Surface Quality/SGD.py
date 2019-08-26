import date
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(date.X, date.Y)
Y_pred =sgd.predict(date.X_test)
acc_sgd = round(sgd.score(date.X_test, date.Y_test)*100, 2)
print("SGD分类器:", acc_sgd)
