import date
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(date.X, date.Y)
Y_pred = decision_tree.predict(date.X_test)
acc_decision_tree = round(decision_tree.score(date.X_test, date.Y_test)*100, 2)
print("决策树:",acc_decision_tree)