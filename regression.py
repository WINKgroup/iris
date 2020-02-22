from bokeh.sampledata.iris import flowers as df
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['petal_length', 'sepal_length']]
y = df['petal_width']

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=.25)
reg1 = LinearRegression().fit(X1_train, y1_train)
print('Performance train set:', reg1.score(X1_train, y1_train), 'test set:', reg1.score(X1_test, y1_test))
print('Coeff:', reg1.coef_, 'Intercept:', reg1.intercept_)

X2_train, X2_test, y2_train, y2_test = train_test_split([[val] for val in X['petal_length']], y, test_size=.25)
reg2 = LinearRegression().fit(X2_train, y2_train)
print('Performance train set:', reg2.score(X2_train, y2_train), 'test set:', reg2.score(X2_test, y2_test))
print('Coeff:', reg2.coef_, 'Intercept:', reg2.intercept_)

df = pd.get_dummies(df)
cols = ['petal_length', 'sepal_length']
cols += list(df.columns[4:])
X3 = df[cols]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=.25)
reg3 = LinearRegression().fit(X3_train, y3_train)
print('Performance train set:', reg3.score(X3_train, y3_train), 'test set:', reg3.score(X3_test, y3_test))
print('Coeff:', reg3.coef_, 'Intercept:', reg3.intercept_)