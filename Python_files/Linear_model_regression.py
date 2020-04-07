""" If any version mismatch error occurs refer this link: https://stackoverflow.com/questions/59474533/modulenotfounderror-no-module-named-numpy-testing-nosetester"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumptionCo2.csv")

# take a look at the dataset
df.head()

print(df.head())

# summarize the data
df.describe()

print(df.describe())

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head(9)

print(cdf.head(9))

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show(block=False)

plt.pause(1)

plt.close()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show(block=False)

plt.pause(1)

plt.close()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show(block=False)

plt.pause(1)

plt.close()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color = 'red')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.title("No of cylinder vs Emission")
plt.show(block=False)
plt.pause(1)

plt.close()

msk = np.random.rand(len(df)) < 0.8
c1 = 0
c2 = 0
for selection_state in msk:
    if selection_state == True:
        c1 += 1
    else:
        c2 +=1
print(f"No of data - Total {len(msk)}, training set {c1}, testing set {c2}\n")   
train = cdf[msk]
print(train)
test = cdf[~msk]
print(test)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show(block=False)
plt.pause(1)

plt.close()

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show(block=False)

plt.pause(1)


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )








