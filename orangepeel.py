import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
import itertools
from matplotlib import animation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_classification
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

df = pd.read_csv("orangepeel.csv",encoding="UTF-8")
F = ["RunLengthNonUniformityNormalized",
"ShortRunEmphasis",
"DependenceVariance",
"RunLengthNonUniformity",
"SmallDependenceEmphasis",
"Correlation",
"DifferenceVariance",
"Contrast",
"DifferenceAverage",
"ZonePercentage",
"Idn",
"Idmn",
"DifferenceEntropy",
"SizeZoneNonUniformity",
"Idm",
"GrayLevelNonUniformity.1",
"Id",
"RunPercentage",
"MCC",
"LargeDependenceEmphasis",
"GrayLevelNonUniformity",
"Imc1",
"InverseVariance",
"DependenceNonUniformity",
"DependenceNonUniformityNormalized",
"LongRunLowGrayLevelEmphasis",
"Complexity",
"LargeAreaLowGrayLevelEmphasis",
"RunEntropy",
"SmallDependenceHighGrayLevelEmphasis"]
features_30 = ["RunLengthNonUniformityNormalized",
"ShortRunEmphasis",
"DependenceVariance",
"RunLengthNonUniformity",
"SmallDependenceEmphasis",
"Correlation",
"DifferenceVariance",
"Contrast",
"DifferenceAverage",
"ZonePercentage",
"Idn",
"Idmn",
"DifferenceEntropy",
"SizeZoneNonUniformity",
"Idm",
"GrayLevelNonUniformity.1",
"Id",
"RunPercentage",
"MCC",
"LargeDependenceEmphasis",
"GrayLevelNonUniformity",
"Imc1",
"InverseVariance",
"DependenceNonUniformity",
"DependenceNonUniformityNormalized",
"LongRunLowGrayLevelEmphasis",
"Complexity",
"LargeAreaLowGrayLevelEmphasis",
"RunEntropy",
"SmallDependenceHighGrayLevelEmphasis"]

df['mode'] = 0
for i in range(df.shape[0]):
  if i % 10 == 8 or i % 10 == 9:
    df['mode'][i] = 1
mask = df['mode'] == 1
mask_2 = df['mode'] == 0
train, test = df[mask_2], df[mask]

df = pd.read_csv("output.csv",encoding="UTF-8")

test_X = df[features_30] 
train_X =  train[features_30]
train_y =  train['L'].values

clf = AdaBoostRegressor( DecisionTreeRegressor(max_depth=4),
                              n_estimators=1000, learning_rate=0.8)
clf.fit(train_X, train_y)

# 預測
test_y_predicted = clf.predict(test_X)

# 績效
#accuracy = mean_squared_error(test_y, test_y_predicted) ** 0.5
#print(accuracy)

sum(test_y_predicted)