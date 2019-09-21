import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("HAM10000_metadata.csv") 
df = df.drop('image_id', axis=1)
df['sex'] = pd.factorize(df['sex'])[0] + 1    # male-1, female-2
df['localization'] = pd.factorize(df['localization'])[0] + 1 
df['dx'] = pd.factorize(df['dx'])[0] + 1 

X = df.drop(columns = ['dx'])
y = df['dx']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 50)}
knn_gs = GridSearchCV(knn, params_knn, cv=3)
knn_gs.fit(X_train, y_train)
knn_best = knn_gs.best_estimator_


rf = RandomForestClassifier()
params_rf = {'n_estimators': [10,25,50,75, 100,125,150,175,200]}
rf_gs = GridSearchCV(rf, params_rf, cv=3)
rf_gs.fit(X_train, y_train)
rf_best = rf_gs.best_estimator_


clf = SVC(gamma='auto')
clf.fit(X_train, y_train)


lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
lr.fit(X_train, y_train)


gnb = GaussianNB()
gnb.fit(X_train, y_train)


print("knn: {}".format(knn_best.score(X_test, y_test)))
print("RandomForest: {}".format(rf_best.score(X_test, y_test)))
print("SVM: {}".format(clf.score(X_test, y_test)))
print("LogisticRegression: {}".format(lr.score(X_test, y_test)))
print("GaussianNB: {}".format(gnb.score(X_test, y_test)))


estimators=[('knn', knn_best), ('RandomForest', rf_best), ('SVM', clf),
			 ('LogisticRegression', lr), ('GaussianNB', gnb)]
ensemble = VotingClassifier(estimators, voting='hard')
ensemble.fit(X_train, y_train)
score = ensemble.score(X_test, y_test)
print("-------------------------")
print("ensemble score: "+str(score))