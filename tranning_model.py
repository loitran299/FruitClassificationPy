import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

labels = ['ca chua', 'cam', 'chanh', 'cherry', 'dua luoi', 'man', 'mo', 'nho', 'quyt', 'tao']
# labels = ['cachua', 'cam', 'chuoi', 'duahau', 'le', 'nho', 'quyt', 'tao', 'thom', 'xoai']

rfc = RandomForestClassifier(random_state=42)

rfc_params = {'n_estimators' : [50, 100, 150, 200, 250, 300, 350, 400],
              'max_depth' : [5, 7, 9, 11, 13, 15, 17, 19]}


edge_cluster_train = pd.read_csv('csv/edge+cluster_valid_features.csv')
# edge_cluster_test = pd.read_csv('/content/drive/My Drive/Fruit recognition/output_input/25_edge+cluster_valid_features.csv')

cluster_features_2_train = pd.read_csv('csv/color_features_train.csv')
# cluster_features_2_test = pd.read_csv('/content/drive/My Drive/Fruit recognition/output_input/25_cluster_features_2_test.csv')


# Ghép 2 dataframe
df_train = pd.concat([cluster_features_2_train, edge_cluster_train.drop(['name'], axis=1)],axis=1)
# df_test = pd.concat([cluster_features_2_test, edge_cluster_test.drop(['name'], axis=1)],axis=1)

Y_train = df_train['name']
X_train = df_train.drop(['name'], axis=1)
# Y_test = df_test['name']
# X_test = df_test.drop(['name'], axis=1)

# Xử lý NaN
X_train = X_train.fillna(X_train.mean())

#Sử dụng GridSearchCv train model RandomForestClassifier
rfc_final = GridSearchCV(rfc, param_grid=rfc_params, n_jobs=-1)
rfc_final.fit(X_train, Y_train)

print('best param:', rfc_final.best_params_)
# print('train_acc:',accuracy_score(Y_train, rfc_final.predict(X_train)))
# rfc_test_acc = accuracy_score(Y_test, rfc_final.predict(X_test))
# print('test acc:', rfc_test_acc)

# Lưu model
rfc_pickle = f"rfc_final_model"
# rfc_pickle = f"rfc_final_model_acc_{round(rfc_test_acc, 3)}"
pickle.dump(rfc_final, open(rfc_pickle, 'wb'))

