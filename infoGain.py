import sys
import cv2
import numpy as np
import os
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, losses

from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost as xgb
import random
import pandas as pd
import sklearn.utils.validation
# read data
from sklearn.model_selection import train_test_split


datainput_gene = pd.read_csv('tempfolder/SelectedGenesExpressionFeatuers2.csv', delimiter=',')
datatarget= pd.read_csv('tempfolder/SelectedClasses.csv', delimiter=',') 

print(datainput_gene.shape)
print(datatarget.shape)
np.random.seed(42)
N = len(datainput_gene)
print(" N is "+str(N));
indices = np.arange(N)
np.random.shuffle(indices)
train_size = int(0.7 * N)
val_size = int(0.15 * N)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

X_train_g = datainput_gene.iloc[train_indices]
y_train      = datatarget.iloc[train_indices]

print(X_train_g.head())

mutual_info = mutual_info_classif(X_train_g, y_train.values.ravel())
print(mutual_info)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train_g.columns
print((mutual_info.sort_values(ascending=False)))
mutual_info.sort_values(ascending=False).to_csv("tempfolder/mutual_info.csv")



