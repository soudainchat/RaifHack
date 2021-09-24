import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, fKFold
from sklearn import ensemble


def data_read(file, sep=None):
    data = pd.read_csv(file, sep=sep)
    return data

def check_data_info(data):
    nans = data.isnull().values.any()
    descr = data.describe()
    return nans, descr

def features_label(data):
    features = data[0:-1]
    label = data[-1]
    return features, label

def train_test(features, label, test=None):
    if test:
        pass
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, label, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def kfold_cv(data, k=2):
    kf = KFold(n_splits=k)
    cv_clf = ensemble.GradientBoostingClassifier(**params)
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in kf.split(X_train, y_train):
        cv_clf.fit(X_train[train], y_train[train])
        val_scores += heldout_score(cv_clf, X_train[test], y_train[test])
    val_scores /= n_splits
    return val_scores


params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'random_state': 3}

reg_ensemble = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)

print("Accuracy: {:.4f}".format(acc))
