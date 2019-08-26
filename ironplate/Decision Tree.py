
from sklearn.tree import DecisionTreeClassifier
import data

decision_tree = DecisionTreeClassifier()
decision_tree.fit(data.X_data, data.y_data)
Y_pred = decision_tree.predict(data.X_data1)
acc_decision_tree = round(decision_tree.score(data.X_data1, data.y_data1)*100, 2)
print("决策树:",acc_decision_tree)
#83.33