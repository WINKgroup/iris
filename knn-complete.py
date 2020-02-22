import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
iris = datasets.load_iris()

# with default hypervalues
pipe = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print('Default Hypervalues score: ', pipe.score(X_test, y_test))
print('Classification report')
print(classification_report(y_test,y_pred))

# with default hypervalues
pipe2 = Pipeline([('scale', StandardScaler()), ('knn', KNeighborsClassifier())])
parameters = {"knn__n_neighbors": np.arange(3, 50)}
cv = GridSearchCV(pipe2, param_grid=parameters, cv=5, iid=False)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print('GridSearchCV score: ', cv.score(X_test,y_test))
print('Classification report')
print(classification_report(y_test,y_pred))
print('Best Hyperparams', cv.best_params_)