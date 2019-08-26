import pandas as pd
import numpy as np
from sklearn.svm import SVC
import date

svc = SVC()
svc.fit(date.X, date.Y)
pred = svc.predict(date.X_test)
print(pred)
acc_scv = round(svc.score(date.X_test, date.Y_test) * 100, 2)
print("svm精确度：", acc_scv)
