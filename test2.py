import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('mpg.csv')
df = df.replace('?', np.nan)
df = df.dropna()
df = df.drop(['name','origin','model_year'], axis=1)
X = df.drop('mpg', axis=1)
y = df[['mpg']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
reg = LinearRegression()
reg.fit(X_train[['horsepower']], y_train)
y_predicted = reg.predict(X_test[['horsepower']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

reg = LinearRegression()
reg.fit(X_train[['horsepower','weight','cylinders']], y_train)
y_predicted = reg.predict(X_test[['horsepower','weight','cylinders']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

print(y_test.shape)
print(y_predicted.shape)
fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()