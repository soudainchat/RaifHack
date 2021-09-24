import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn import ensemble
import scipy
from pandas.api.types import is_object_dtype
from metrics import metrics_stat


def data_read(file, sep=None):
    data = pd.read_csv(file, sep=sep)
    return data

def check_data_info(data):
    descr = data.describe()
    info = data.info()
    return descr, info

def fill_nans(data):
    for col in data.columns:
        if data[col].isnull().values.any() == True:
            if data[col].dtype == 'float64':
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            pass
    return data

def features_label(data, label):
    features = data.drop(label,  axis=1)
    label = data[label]
    return features, label

def train_test(features, label):
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


train = data_read("data/train.csv")
test = data_read("data/test.csv")


train = fill_nans(train)

train = train.drop(["street","floor", "id", "city", "date", "osm_city_nearest_name","region"], axis=1)

test = test.drop(["street","floor", "id", "city", "date", "osm_city_nearest_name","region"], axis=1)

features, label = features_label(train, 'per_square_meter_price')
X_train, X_test, y_train, y_test = train_test(features, label)

params = {'n_estimators': 3000, 'learning_rate': 0.001, 'random_state': 3, 'verbose':250}
model = CatBoostRegressor(**params)
model.fit(features, label)

prediction = model.predict(test)


solution = test_1[['id']].copy()
solution['per_square_meter_price'] = pd.Series(np.full(len(test_1), prediction))
solution.to_csv('first.csv', sep=',', index=False)