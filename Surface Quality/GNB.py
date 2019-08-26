import date
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(date.X, date.Y)
label_pred = gaussian.predict(date.X_test)
acc_gaussian = round(gaussian.score(date.X_test, date.Y_test)*100, 3)
print("朴素贝叶斯:", acc_gaussian)