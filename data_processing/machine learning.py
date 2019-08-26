import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from Gray_level_co_occurrence_matrix import *

imgpath = './img/IMG20190608115455.jpg'

newpath = './img/new'
cut(imgpath, 5, newpath)
print(newpath+imgpath[5:])

result = [str(item) for item in test(newpath+imgpath[6:])]
path='./new/new.csv'  #give path where extracted features are saved

abc=pd.read_csv(path)

X=np.array((abc.as_matrix())[1:,1:])
Y=X[:,4]
X=X[:,0:4]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
y_train = y_train*10
y_test = y_test*10
y_train = y_train.astype('int')
y_test = y_test.astype('int')
model=KNeighborsClassifier()

K_value = 2
neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train)
testt=neigh.predict(X_test)
print(testt)
scoree=accuracy_score(y_test, testt)
print("KNN testing accuracy=",scoree*100)

X1=np.array(result)
X1 = X1.reshape(1,-1)
testt = neigh.predict(X1)
testt = testt/10
print(testt)
