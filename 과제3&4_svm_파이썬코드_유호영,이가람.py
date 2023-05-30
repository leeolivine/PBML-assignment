'''
# 과제 3&4 SVM, 18011152 유호영 / 21011215 이가람

"Generating Datasets using Scikit-learn"

Assignment 3&4 - SVM

@author: You Ho Yeong, Lee Ga Ram
'''

#%% Import Library
import mglearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

#%% Cluster List
cluster_num_list = [100, 500, 1000]
cluster_std_list = [1, 2, 3]
cluster_cen_list = [2, 5, 10]
#%% Changing N_samples
for n_samples in cluster_num_list:
    X, y, center = make_blobs(n_samples=n_samples, n_features=2, centers=5, cluster_std=1,
                               random_state=100, return_centers=True)
    palette = sns.hls_palette(len(center))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, legend='full')
    plt.scatter(center[:, 0], center[:, 1], c='black', s=200, 
                alpha=1, marker='+', zorder = 10)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()

    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()
    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/Datasets_num{n_samples}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()

    # Logistic - Train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2, 
                                                        random_state=100)

    # Logistic - Data Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('y_test.shape: ', y_test.shape)

    # Logistic - HyperParameter-Tuning
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']} 
    kfold = KFold(n_splits=10, shuffle=True, random_state=100)
    grid = GridSearchCV(SVC(random_state=100), param_grid, cv=kfold, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("\nBest score: ", grid.best_score_)
    print("\nBest params: ", grid.best_params_)

    # Logistic - Fitting using best model
    classifier = grid.best_estimator_
    classifier.fit(X_train, y_train)

    # Logistic - Model Evaluation
    y_pred = classifier.predict(X_test)

    print('Report :\n')
    print(classification_report(y_test,y_pred))
    print('Accuracy Score: ', accuracy_score(y_pred,y_test))
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

    # Logistic - Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix Samples={n_samples}')
    plt.tight_layout()
    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/CF_num{n_samples}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()

    # Logistic - Plotting Result
    # Plotting Actual classes vs Predicted classes
    plt.figure(figsize=(14,7))

    plt.subplot(1,2,1)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette=sns.hls_palette(len(center)), legend='full')
    plt.title(f'Actual Classes: y_test, Samples = {n_samples}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()

    plt.subplot(1,2,2)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, palette=sns.hls_palette(len(center)), legend='full')
    plt.title(f'Predicted Classes: y_pred, Samples = {n_samples}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()

    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/Result_num{n_samples}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()

#%% Changing cluster_std
for cluster_std in cluster_std_list:
    X, y, center = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=cluster_std,
                               random_state=100, return_centers=True)
    palette = sns.hls_palette(len(center))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, legend='full')
    plt.scatter(center[:, 0], center[:, 1], c='black', s=200, 
                alpha=1, marker='+', zorder = 10)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()

    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()
    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/Datasets_std{cluster_std}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()

    # Logistic - Train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2, 
                                                        random_state=100)

    # Logistic - Data Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('y_test.shape: ', y_test.shape)

    # Logistic - HyperParameter-Tuning
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']} 
    kfold = KFold(n_splits=10, shuffle=True, random_state=100)
    grid = GridSearchCV(SVC(random_state=100), param_grid, cv=kfold, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("\nBest score: ", grid.best_score_)
    print("\nBest params: ", grid.best_params_)

    # Logistic - Fitting using best model
    classifier = grid.best_estimator_
    classifier.fit(X_train, y_train)

    # Logistic - Model Evaluation
    y_pred = classifier.predict(X_test)

    print('Report :\n')
    print(classification_report(y_test,y_pred))
    print('Accuracy Score: ', accuracy_score(y_pred,y_test))
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

    # Logistic - Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix STD={cluster_std}')
    plt.tight_layout()
    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/CF_std{cluster_std}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()

    # Logistic - Plotting Result
    # Plotting Actual classes vs Predicted classes
    plt.figure(figsize=(14,7))

    plt.subplot(1,2,1)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette=sns.hls_palette(len(center)), legend='full')
    plt.title(f'Actual Classes: y_test, STD = {cluster_std}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()

    plt.subplot(1,2,2)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, palette=sns.hls_palette(len(center)), legend='full')
    plt.title(f'Predicted Classes: y_pred, STD = {cluster_std}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()

    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/Result_std{cluster_std}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()


#%% Changing centers
for centers in cluster_cen_list:
    X, y, center = make_blobs(n_samples=500, n_features=2, centers=centers, cluster_std=1,
                               random_state=100, return_centers=True)
    palette = sns.hls_palette(len(center))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, legend='full')
    plt.scatter(center[:, 0], center[:, 1], c='black', s=200, 
                alpha=1, marker='+', zorder = 10)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()

    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()
    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/Datasets_cen{centers}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()

    # Logistic - Train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2, 
                                                        random_state=100)

    # Logistic - Data Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('y_test.shape: ', y_test.shape)

    # Logistic - HyperParameter-Tuning
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']} 
    kfold = KFold(n_splits=10, shuffle=True, random_state=100)
    grid = GridSearchCV(SVC(random_state=100), param_grid, cv=kfold, scoring='accuracy')
    grid.fit(X_train, y_train)
    print("\nBest score: ", grid.best_score_)
    print("\nBest params: ", grid.best_params_)

    # Logistic - Fitting using best model
    classifier = grid.best_estimator_
    classifier.fit(X_train, y_train)

    # Logistic - Model Evaluation
    y_pred = classifier.predict(X_test)

    print('Report :\n')
    print(classification_report(y_test,y_pred))
    print('Accuracy Score: ', accuracy_score(y_pred,y_test))
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

    # Logistic - Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix Center={centers}')
    plt.tight_layout()
    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/CF_cen{centers}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()

    # Logistic - Plotting Result
    # Plotting Actual classes vs Predicted classes
    plt.figure(figsize=(14,7))

    plt.subplot(1,2,1)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, palette=sns.hls_palette(len(center)), legend='full')
    plt.title(f'Actual Classes: y_test, Centers = {centers}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()

    plt.subplot(1,2,2)
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, palette=sns.hls_palette(len(center)), legend='full')
    plt.title(f'Predicted Classes: y_pred, Centers = {centers}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    ax = plt.gca()
    handles, _ = ax.get_legend_handles_labels()
    labels = [f'Class {i}' for i in range(0, len(center))]
    plt.legend(handles, labels)
    plt.tight_layout()

    out_path = f'/Users/hoyeong/Desktop/python/PBML/HW/Assign3_4/Image/Result_cen{centers}'
    plt.savefig(out_path, dpi=600)
    plt.ion()
    plt.show()
    plt.close()