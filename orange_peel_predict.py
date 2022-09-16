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