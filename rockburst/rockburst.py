#!~/anaconda3/bin/python3.6
# encoding: utf-8

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.utils import shuffle

from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA


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

# norm
# x = preprocessing.scale(x)

# ==========================================================
scaler = RobustScaler(with_centering=False)
x = scaler.fit_transform(x)

# normalizer = MinMaxScaler()
# x = normalizer.fit_transform(x.astype(float))

pca = PCA(0.99, random_state=0)
x = pca.fit_transform(x)
# ==========================================================

y = y.flatten()
# x.shape=(150,4)   y.shape=(150,)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)


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
