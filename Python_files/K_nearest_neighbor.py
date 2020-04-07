import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df = pd.read_csv("teleCust1000t.csv")

# print(df.head())

# print(df['custcat'].value_counts())

# df.hist(column='income', bins=50)
# plt.show()

# print(df.columns)

##To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:


X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values  #.astype(float)

# print(X[0:5])

y = df['custcat'].values

# print(y[0:5])

# print(X[0:5])

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# print(X[0:5])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# print ('Train set:', X_train.shape,  y_train.shape)

# print ('Test set:', X_test.shape,  y_test.shape)


from sklearn.neighbors import KNeighborsClassifier

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# print(neigh)

yhat = neigh.predict(X_test)
# print(yhat[0:5])

# print(y[0:5])

from sklearn import metrics
# print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# Learners code section

Accuracy = []

for i in range(1,80):
   
    neighbor = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)

    yhat = neighbor.predict(X_test)

    Accuracy.append([metrics.accuracy_score(y_test, yhat), metrics.accuracy_score(y_train, neigh.predict(X_train)),i])

Accuracy.sort(reverse = True)

print(f" Maximum accuacry is achieved using K = {Accuracy[0][2]}, Test accuracy = {Accuracy[0][0]}, Train accuracy = {Accuracy[0][1]} ")





