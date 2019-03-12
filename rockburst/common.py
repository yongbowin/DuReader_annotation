#!~/anaconda3/bin/python3.6
# encoding: utf-8

import numpy as np
import pandas as pd

from sklearn import preprocessing

# for SVM Classifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.utils import shuffle

from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA

# for KNN Classifier
from sklearn import neighbors
n_neighbors = 15

# for RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

# for xgboost Classifier
from xgboost import XGBClassifier

# for GradientBoosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# for DecisionTree Classifier
from sklearn.tree import DecisionTreeClassifier

# for ExtraTrees Classifier
from sklearn.ensemble import ExtraTreesClassifier

# for AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

# MLP Classifier
from sklearn.neural_network import MLPClassifier

BASE_PATH = '/home/wyb/PycharmProjects/DuReader_annotation/rockburst/'

# read CSV
csv_data = pd.read_csv(BASE_PATH + 'data_nofill.csv')

# label: L, M, N, H
# print(csv_data["label"].count())  # nums=246

# calculate each classification nums
l_count = 0
m_count = 0
n_count = 0
h_count = 0
for i in csv_data["label"]:
    if i == "L":
        l_count += 1
    elif i == "M":
        m_count += 1
    elif i == "N":
        n_count += 1
    elif i == "H":
        h_count += 1

print("L nums: ", l_count)  # 78
print("M nums: ", m_count)  # 81
print("N nums: ", n_count)  # 43
print("H nums: ", h_count)  # 44

# check the NAN value
x = csv_data.isnull().sum()
# print(x)

# fill the NAN value with mean of corresponding classification
l_count = 0
m_count = 0
n_count = 0
h_count = 0
l_sum = 0
m_sum = 0
n_sum = 0
h_sum = 0
for i in csv_data["label"]:
    if i == "L":
        l_count += 1
        l_sum += 1
    elif i == "M":
        m_count += 1
    elif i == "N":
        n_count += 1
    elif i == "H":
        h_count += 1


# calculate mean of each classification to fill the NAN
mean_series = csv_data["feature1"].groupby(csv_data["label"]).mean()
l_mean = mean_series["L"]
m_mean = mean_series["M"]
n_mean = mean_series["N"]
h_mean = mean_series["H"]


for name, group in csv_data.groupby(csv_data["label"]):
    if name == "L":
        l_group = group.fillna(l_mean)
    elif name =="M":
        m_group = group.fillna(m_mean)
    elif name == "N":
        n_group = group.fillna(n_mean)
    elif name == "H":
        h_group = group.fillna(h_mean)


# csv_data_filled = pd.concat([l_group, m_group, n_group, h_group], axis=0, ignore_index=True)
csv_data_filled = pd.concat([l_group, m_group, n_group, h_group, n_group, h_group], axis=0, ignore_index=True)
csv_data_filled = shuffle(csv_data_filled)

# csv_data_filled.to_csv(BASE_PATH + 'data_filled_add_sample_shuffle.csv', index=False)


def label_type(s):
    # bytes to str
    s = str(s, encoding="utf8")
    it = {'L': 0, 'M': 1, 'N': 2, 'H': 3}
    return it[s]


path = BASE_PATH + 'data_filled_add_sample_shuffle.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={8: label_type})

x, y = np.split(data, (8,), axis=1)

# # norm
# x = preprocessing.scale(x)

# Robust Scaler
scaler = RobustScaler(with_centering=False)
x = scaler.fit_transform(x)

# # MinMax Scaler
# normalizer = MinMaxScaler()
# x = normalizer.fit_transform(x.astype(float))

# pca
pca = PCA(0.99, random_state=0)
x = pca.fit_transform(x)

y = y.flatten()
# x.shape=(150,4)   y.shape=(150,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)


def svm_pred(x_train, x_test, y_train, y_test):
    """
    run with SVM
    """
    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])
    parameters = {
        'clf__gamma': (0.005, 0.01, 0.03, 0.1, 0.3, 1, 1.3),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30, 40, 50, 60),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', refit=True)
    grid_search.fit(x_train, y_train)

    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数集：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    predictions = grid_search.predict(x_test)
    print("\n")
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))


def knn_pred(x_train, x_test, y_train, y_test):
    """
    run with KNN
    """
    max_val = 0
    best_k = 0
    best_weights = ''
    for i in range(20):
        n_neighbors = i + 1
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(x_train, y_train)

            predictions = clf.predict(x_test)

            acc = accuracy_score(y_test, predictions)
            if acc > max_val:
                max_val = acc
                best_k = n_neighbors
                best_weights = weights
            # print("n_neighbors=" + str(n_neighbors) + ", When weights is " + weights + ", the acc is " + str(acc))

    print("Best weights=" + best_weights + ", Best K=" + str(best_k) + ", Max acc=" + str(max_val))


# knn_pred(x_train, x_test, y_train, y_test)


def rf_pred(x_train, x_test, y_train, y_test):
    """
    run with random forest
    """
    rfc = RandomForestClassifier(random_state=0, n_jobs=3)
    rfc.fit(x_train, y_train)
    predictions = rfc.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    print("rfc acc: ", acc)

    # print('The accuracy of RFC on testing set:', rfc.score(x_test, y_test))


# rf_pred(x_train, x_test, y_train, y_test)


def xgbc_pred(x_train, x_test, y_train, y_test):
    """
    run xgboost

    params:
        XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                   gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
                   min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                   objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                   scale_pos_weight=1, seed=0, silent=True, subsample=1)
    """
    nestimators = []
    maxdepth = []
    for i in range(100, 1100, 200):
        nestimators.append(i)
    for j in range(2, 7):
        maxdepth.append(j)

    parameters = {
        'max_depth': maxdepth,
        'n_estimators': nestimators,
        'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]
    }

    xgbc_best = XGBClassifier()
    grid_search = GridSearchCV(xgbc_best, parameters, n_jobs=-1, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(x_train, y_train)

    print('最佳效果：%0.3f' % grid_search.best_score_)
    print('最优参数集：')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    predictions = grid_search.predict(x_test)
    print("\n")
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))


# xgbc_pred(x_train, x_test, y_train, y_test)


def gb_pred(x_train, x_test, y_train, y_test):
    """
    run GradientBoosting Classifier
    """
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)
    predictions = clf.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    print("GradientBoostingClassifier acc: ", acc)


# gb_pred(x_train, x_test, y_train, y_test)


def dt_pred(x_train, x_test, y_train, y_test):
    """
    run DecisionTree Classifier
    """
    clf = DecisionTreeClassifier(max_depth=None, random_state=0).fit(x_train, y_train)
    predictions = clf.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    print("GradientBoostingClassifier acc: ", acc)


# dt_pred(x_train, x_test, y_train, y_test)


def et_pred(x_train, x_test, y_train, y_test):
    """
    run ExtraTrees Classifier
    """
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, random_state=0).fit(x_train, y_train)
    predictions = clf.predict(x_test)

    acc = accuracy_score(y_test, predictions)
    print("GradientBoostingClassifier acc: ", acc)


# et_pred(x_train, x_test, y_train, y_test)


def adaboost_pred(x_train, x_test, y_train, y_test):
    """
    run AdaBoost Classifier
    """
    bdt_discrete = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=1.5,
        algorithm="SAMME")
    bdt_discrete.fit(x_train, y_train)
    bdt_discrete_predictions = bdt_discrete.predict(x_test)

    bdt_discrete_acc = accuracy_score(y_test, bdt_discrete_predictions)
    print("bdt_discrete_acc acc: ", bdt_discrete_acc)


# adaboost_pred(x_train, x_test, y_train, y_test)


def nn_pred(x_train, x_test, y_train, y_test):
    """
    run MLP Classifier

    nn best acc:  0.85 , best larer1 size:  90 , best larer2 size:  45 , best_layer3_size:  70
    """
    layer_size_list = []
    for hidden_size in range(5, 200, 5):
        layer_size_list.append(hidden_size)
    # layer_size_list = [130, 120, 110, 100, 90, 80, 70, 60, 55, 50, 45, 40, 30, 20, 10, 5]
    best_layer1_size = 100
    best_layer2_size = 100
    best_layer3_size = 100
    best_layer4_size = 100
    best_acc = 0

    for i in layer_size_list:
        for j in layer_size_list:
            for k in layer_size_list:
                for m in layer_size_list:
                    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i, j, k, m), random_state=1)
                    clf.fit(x_train, y_train)
                    predictions = clf.predict(x_test)

                    acc = accuracy_score(y_test, predictions)
                    if acc > best_acc:
                        best_acc = acc
                        best_layer1_size = i
                        best_layer2_size = j
                        best_layer3_size = k
                        best_layer4_size = m

    print("nn best acc: ", best_acc,
          ", best larer1 size: ", best_layer1_size,
          ", best larer2 size: ", best_layer2_size,
          ", best_layer3_size: ", best_layer3_size,
          ", best_layer4_size: ", best_layer4_size
          )


nn_pred(x_train, x_test, y_train, y_test)

