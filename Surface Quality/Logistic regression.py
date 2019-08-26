import date
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(date.X, date.Y)  # 训练数据
print('逻辑回归训练集准确率：\n', LR.score(date.X, date.Y))
print('逻辑回归验证集准确率：\n', LR.score(date.X_test, date.Y_test))

