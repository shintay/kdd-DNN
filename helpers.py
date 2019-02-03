#%%
import h5py
import pandas as pd
import numpy as np

from sklearn.preprocessing import (Normalizer,
                                   StandardScaler,
                                   MinMaxScaler,
                                   OneHotEncoder)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             accuracy_score,
                             mean_squared_error,
                             mean_absolute_error)

#%%
### NSL-KDD DataSet ###
TRAIN = r'C:/Users/thiago/Development/udacity/06-Capstone/NSL-KDD/df_train_probe.csv'
TEST = r'C:/Users/thiago/Development/udacity/06-Capstone/NSL-KDD/df_test_probe.csv'
TARGET_TRAIN = r'C:/Users/thiago/Development/udacity/06-Capstone/NSL-KDD/target_train_probe.csv'
TARGET_TEST = r'C:/Users/thiago/Development/udacity/06-Capstone/NSL-KDD/target_test_probe.csv'


# import dataset addind columns / features names
df_train_probe = pd.read_csv(TRAIN)
df_test_probe = pd.read_csv(TEST)
target_train_probe = pd.read_csv(TARGET_TRAIN)
target_test_probe = pd.read_csv(TARGET_TEST)

# norm data
X = df_train_probe
y = target_train_probe
X_train_probe = np.array(Normalizer().fit_transform(X))
y_train_probe = np.array(y)

X = df_test_probe
y = target_test_probe
X_test_probe = np.array(Normalizer().fit_transform(X))
y_test_probe = np.array(y)

# gridsearch for decisiontrees
model_dt = DecisionTreeClassifier()

parameters = {'max_depth': range(2,20,2),
              'min_samples_leaf': range(2,20,2),
              'min_samples_split': range(2,20,2)}

scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(model_dt,
                        param_grid = parameters,
                        scoring = scorer)

grid_fit = grid_obj.fit(X_train_probe, y_train_probe)
best_model_dt = grid_fit.best_estimator_
best_model_dt.fit(X_train_probe, y_train_probe)
y_pred_dt = best_model_dt.predict(X_test_probe)
print(accuracy_score(y_test_probe, y_pred_dt))

# # gridsearch for KNN
# parameters = [{'weights': ['uniform', 'distance'],
#                'n_neighbors': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]

# model_knn_cv = GridSearchCV(KNeighborsClassifier(), parameters)
# model_knn_cv.fit(X_train_probe, y_train_probe)
# y_pred_knn_cv = model_knn.predict(X_test_probe)
# print(accuracy_score(y_test_probe, y_pred_knn_cv))


#%%
