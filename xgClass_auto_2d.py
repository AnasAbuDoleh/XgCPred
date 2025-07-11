from sklearn import svm
import sys
import cv2
import numpy as np
import os
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, losses
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost as xgb
import random
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

# read data
from sklearn.model_selection import train_test_split
vImageFeatSize = int(sys.argv[1])
vAutoencEpo = int(sys.argv[2])
gN_estimators=int(sys.argv[3])


basepath = 'tempfolder/outputImages/'
img_array = []
dim=(40,40)
for filename in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, filename)):
        fname = os.path.join(basepath, filename)
        im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        #im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        img_array.append(np.asarray(im))
img_array = np.array(img_array)
print(img_array.shape)
x_train = img_array.astype('float32') / 255.

class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(dim[0], dim[1], 1)),
      layers.Conv2D(4, (3, 3), activation='relu', padding='same', strides=1),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
      #layers.Flatten(),
      #layers.Dense(vImageFeatSize ),
     ])

    self.decoder = tf.keras.Sequential([
      #layers.Dense(units=5 * 5 * 32, activation=tf.nn.relu),
      #layers.Reshape(target_shape=(5, 5, 32)), 
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(4, kernel_size=3, strides=1, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x_train, x_train,
                epochs=vAutoencEpo,
                shuffle=True)

encoded_imgs = autoencoder.encoder(x_train).numpy()
f_encoded_imgs= np.reshape(encoded_imgs, (encoded_imgs.shape[0], encoded_imgs.shape[1]*encoded_imgs.shape[2]*encoded_imgs.shape[3]))

datainput_auto = f_encoded_imgs
datatarget = pd.read_csv('tempfolder/SelectedClasses.csv', delimiter=',')

X_train_g, X_test1_g, y_train_g, y_test1_g = train_test_split(
    datainput_auto, datatarget, test_size=.3, random_state=100)
X_valid_g, X_test_g, y_valid_g, y_test_g = train_test_split(
    X_test1_g, y_test1_g, test_size=.5, random_state=120)

y_train_g = y_train_g.values.ravel()
y_valid_g = y_valid_g.values.ravel()
y_test_g = y_test_g.values.ravel()

print(X_train_g.shape)

model_g = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=gN_estimators,
                        objective='multi:softmax', num_class=10, booster='gbtree', verbosity=1,
                        eval_metric='merror', tree_method="hist", enable_categorical=True)
model_g.fit(X_train_g, y_train_g, eval_set=[(X_valid_g, y_valid_g)])
y_pred_g = model_g.predict_proba(X_valid_g)
y_pred_gX = model_g.predict(X_valid_g)
accuracy_g = accuracy_score(y_valid_g, y_pred_gX)
print(accuracy_g)
y_pred_gXtest = model_g.predict(X_test_g)
accuracy_gtest = accuracy_score(y_test_g, y_pred_gXtest)
print(accuracy_gtest)
f1m = f1_score(y_test_g, y_pred_gXtest, average='macro')
f1w = f1_score(y_test_g, y_pred_gXtest, average='weighted')
# print(accuracy)
print(f1m)
print(f1w)

