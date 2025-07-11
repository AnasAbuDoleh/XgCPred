from sklearn import svm
import sys
import cv2
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import tensorflow as tf
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xgboost as xgb
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# read data
from sklearn.model_selection import train_test_split
vImageFeatSize = int(sys.argv[1])
vAutoencEpo = int(sys.argv[2])
vGeneFeatSize = int(sys.argv[3])
vGene_estimators=int(sys.argv[4])
vSparseFract=int(sys.argv[5])
#vFinalEpo=int(sys.argv[6])
#vRandSeed=int(sys.argv[7])
dropout_rate=0.1


datatarget1 = pd.read_csv('tempfolder/SelectedClasses.csv', delimiter=',', header=0)
datatarget1 = datatarget1.values.ravel()
datatarget = datatarget1[np.arange(len(datatarget1)) % vSparseFract == 0]
unique_items = np.unique(datatarget)
num_classes = len(unique_items)
print("\n Number of classses "+str(num_classes))
print("\n Number of items "+str(datatarget.shape))
class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(64, 64, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same',strides=1),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',strides=2),
            #layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',strides=2),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(vImageFeatSize, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, x):
        return self.encoder(x)
    
basepath = 'tempfolder/outputImages/'
img_array = []
dim=(64,64)
icount=0
for filename in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, filename)):
        fname = os.path.join(basepath, filename)
        im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if icount % vSparseFract == 0:
          img_array.append(np.asarray(im))
        icount=icount+1
img_array = np.array(img_array)
print("Image dataset size :"+str(img_array.shape))

datainput_imgs = img_array.astype('float32') / 255.
########################################################################################
datainput_genes = pd.read_csv('tempfolder/SelectedGenesExpressionFeatuers.csv', delimiter=',')
datainput_genes = datainput_genes.iloc[:,0:vGeneFeatSize]
#datainput_genes = datainput_genes1[np.arange(len(datainput_genes1)) % vSparseFract == 0]
print(datainput_genes.shape)
#######################################################################################

N = len(datainput_genes)
print(" N is "+str(N));
print(str(N),file=open('results.txt', 'a'))
vRandSeedList = [12,118,420,1000,2330]
vRandSeedList = [10]
for vRandSeed in vRandSeedList:
    np.random.seed(vRandSeed)
    indices = np.arange(N)
    #np.random.shuffle(indices)
    train_size = int(0.8 * N)
    val_size = int(0.10 * N)
    #train_indices = indices[:train_size]
    #val_indices = indices[train_size:train_size + val_size]
    #test_indices = indices[train_size + val_size:]
    
    range1 = np.arange(0,1554+5707)  # From 0 to 4
    range2 = np.arange(10150-1449,10150)  # From 10 to 20
    indicesX = np.concatenate([range1, range2])
    train_indicesX = indices[indicesX]
    train_indices, val_indices = train_test_split(train_indicesX, test_size=0.1, random_state=42)
    test_indices  = indices[1554+5707:1554+5707+1440]  
    print(datatarget.shape)    
    
    X_train_gene = datainput_genes.iloc[train_indices]
    y_train      = datatarget[train_indices]
    X_valid_gene = datainput_genes.iloc[val_indices]
    y_valid      = datatarget[val_indices]
    X_test_gene = datainput_genes.iloc[test_indices]
    y_test      = datatarget[test_indices]
    x_train     = datainput_imgs[train_indices]
    y_trainC     = tf.keras.utils.to_categorical(datatarget[train_indices],num_classes)
    x_valid     = datainput_imgs[val_indices]
    y_validC     = tf.keras.utils.to_categorical(datatarget[val_indices],num_classes)
    x_test      = datainput_imgs[test_indices]
    y_testC      = tf.keras.utils.to_categorical(datatarget[test_indices],num_classes)

    unique_labels = np.unique(datatarget)
    print(" Number of classes is "+str(unique_labels))
    weights = compute_class_weight('balanced', classes=unique_labels, y=datatarget)
    weights_dict = {label: weight for label, weight in zip(unique_labels, weights)}
    dtrain = xgb.DMatrix(X_train_gene, label=y_train, weight=[weights_dict[label] for label in y_train])
    dvalid = xgb.DMatrix(X_valid_gene, label=y_valid)
    dtest =  xgb.DMatrix(X_test_gene, label=y_test)
    #print("\nafter weight "+str(dtrain.shape))

    print(X_train_gene.shape)
    params = {
        'max_depth': 6,
        'learning_rate': 0.3,
        #'n_estimators': vGene_estimators,
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'booster': 'gbtree',
        'verbosity': 1,
        'eval_metric': 'merror',
        'tree_method': "hist",
        #'enable_categorical': True
    }
    model_g = xgb.train(
        params,
        dtrain,
        num_boost_round=vGene_estimators,
        evals=[(dvalid, 'eval')],
        early_stopping_rounds=20
    )
    print("\n............ XGBOOST RESULTS..............\n")
    y_valid_gProb = model_g.predict(dvalid)
    y_valid_pred = np.argmax(y_valid_gProb, axis=1)
    accuracy_gValid = accuracy_score(y_valid, y_valid_pred)
    print(" Valid Accuracy "+str(accuracy_gValid))

    y_test_gProb = model_g.predict(dtest)
    y_test_pred = np.argmax(y_test_gProb, axis=1)
    accuracy_gTest = accuracy_score(y_test, y_test_pred)
    print(" Test Accuracy "+str(accuracy_gTest))

    f1_scores_per_class = f1_score(y_test, y_test_pred, average=None)
    print(f1_scores_per_class.shape)
    #print(f1_scores_per_class)
    median_f1 = np.median(f1_scores_per_class)
    print(" Median F1 "+str(median_f1))
    f1_scores_per_class = f1_score(y_test, y_test_pred, average='weighted')
    #print(f1_scores_per_class.shape)
    #mean_f1 = np.mean(f1_scores_per_class)
    mean_f1 = f1_scores_per_class
    print(" Weighted F1 "+str(mean_f1))
    print(accuracy_gTest,file=open('results.txt', 'a'))
    print(median_f1,file=open('results.txt', 'a'))
    print(mean_f1,file=open('results.txt', 'a'))
    importance = model_g.get_score(importance_type='gain')
    # Sorting the importances
    importances_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    #for feature, importance in importances_sorted:
    #    print(f"Feature: {feature}, Importance: {importance}")

    with open('importance_output.txt', 'w') as file:
        for feature, importance in importances_sorted:
            file.write(f"{feature},{importance}\n")

    #importances = model_g.feature_importances_
    #sorted_indices = np.argsort(importances)[::-1]
    #top_n = 50
    #for i in range(top_n):
    #    print(f"Feature {sorted_indices[i]}: Importance: {importances[sorted_indices[i]]}")
    ####################################

    model = Classifier()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

    model.fit(x_train, y_trainC, epochs=vAutoencEpo, batch_size=256, validation_data=(x_valid, y_validC),callbacks=[early_stopping])
    y_test_AProb = model.predict(x_test)
    y_valid_AProb = model.predict(x_valid)
    print(y_test_AProb.shape)
    loss, accuracy = model.evaluate(x_test, y_testC)
    print("\n............ CNN RESULTS..............\n")
    print(f"Test accuracy: {accuracy*100:.2f}%")

    predicted_classesValid = np.argmax(y_valid_AProb, axis=1)
    accuracyValid = accuracy_score(y_valid, predicted_classesValid)
    print("Valid Accuracy:", accuracyValid)
    predicted_classes = np.argmax(y_test_AProb, axis=1)
    f1_scores_per_class = f1_score(y_test, predicted_classes, average=None)
    #predicted_classes_categorical = to_categorical(predicted_classes, num_classes=num_classes)
    accuracy = accuracy_score(y_test, predicted_classes)
    print("Test Accuracy:", accuracy)
    f1_scores = f1_score(y_test, predicted_classes, average=None)
    median_f1 = np.median(f1_scores)
    print("Median F1 Score:", median_f1)
    f1_scores_per_class = f1_score(y_test, predicted_classes, average='weighted')
    #print(f1_scores_per_class.shape)
    #mean_f1 = np.mean(f1_scores_per_class)
    mean_f1 = f1_scores_per_class
    print(" Weighted F1 "+str(mean_f1))
    #mean_f1 = np.mean(f1_scores)
    #print(" Mean F1 "+str(mean_f1))
    print(accuracy,file=open('results.txt', 'a'))
    print(median_f1,file=open('results.txt', 'a'))
    print(mean_f1,file=open('results.txt', 'a'))

    avgres = (accuracy_gValid*y_test_gProb + accuracyValid*y_test_AProb)/(accuracy_gValid+accuracyValid)
    predicted_classes = np.argmax(avgres, axis=1)
    accuracy = accuracy_score(y_test, predicted_classes)
    print("Accuracy:", accuracy)
    f1_scores = f1_score(y_test, predicted_classes, average=None)
    median_f1 = np.median(f1_scores)
    print("Median F1 Score:", median_f1)
    f1_scores_per_class = f1_score(y_test, predicted_classes, average='weighted')
    #print(f1_scores_per_class.shape)
    #mean_f1 = np.mean(f1_scores_per_class)
    mean_f1 = f1_scores_per_class
    print(" Weighted F1 "+str(mean_f1))
    #mean_f1 = np.mean(f1_scores)
    print(" Mean F1 "+str(mean_f1))
    print(accuracy,file=open('results.txt', 'a'))
    print(median_f1,file=open('results.txt', 'a'))
    print(mean_f1,file=open('results.txt', 'a'))


    
