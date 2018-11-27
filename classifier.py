#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV, learning_curve, ShuffleSplit
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# ### Dataset

# In[ ]:


def getData():
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    train_X, test_X = train_X.reshape(-1, 3072), test_X.reshape(-1, 3072)
    train_y, test_y = train_y.reshape(-1), test_y.reshape(-1)
    return train_X, train_y, test_X, test_y


# In[ ]:


def dispSize(X_train, y_train, X_test, y_test, title):
    print("-"*100)
    print(title)
    print("train_X.shape:", X_train.shape, "test_X.shape: ", X_test.shape)
    print("train_y.shape:", y_train.shape, "test_y.shape: ", y_test.shape)


# ### CIFAR-10 Dataloader

# In[ ]:


train_X, train_y, test_X, test_y = getData()
dispSize(train_X, train_y, test_X, test_y, "Raw Dataset")


# ### Normalization

# In[ ]:


train_X_norm = (train_X - np.mean(train_X, axis=0)) / np.std(train_X, axis=0)
test_X_norm = (train_X - np.mean(test_X, axis=0)) / np.std(test_X, axis=0)
dispSize(train_X_norm, train_y, test_X_norm, test_y, "Normalized Dataset")


# In[ ]:


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


# In[ ]:


cv = ShuffleSplit(n_splits = 6, test_size = 0.2, random_state=0)


# ### PCA

# In[ ]:


def pca(train_X, test_X, num_components=500):
    pca = decomposition.PCA(n_components=num_components)
    pca.fit(train_X)
    new_train_X = pca.transform(train_X)
    new_test_X = pca.transform(test_X)
    return new_train_X, new_test_X


# In[ ]:


train_X_pca1, test_X_pca1 = pca(train_X_norm, test_X_norm, num_components = 350)
train_X_pca, test_X_pca = pca(train_X_norm, test_X_norm, num_components = 15)
# train_X_norm_pca, test_X_norm_pca = pca(train_X_norm, test_X_norm, num_components=350)
dispSize(train_X_pca, train_y, test_X_pca, test_y, "Normalized PCA Dataset")
dispSize(train_X_pca1, train_y, test_X_pca1, test_y, "Normalized PCA Dataset")
# dispSize(train_X_norm_pca, train_y, test_X_norm_pca, test_y, "Normalized PCA Dataset")


# ### LDA

# In[ ]:


def lda(train_X, train_y, test_X):
    lda = LDA()
    lda.fit(train_X, train_y)
    new_train_X = lda.transform(train_X)
    new_test_X = lda.transform(test_X)
    return new_train_X, new_test_X


# In[ ]:


train_X_lda, test_X_lda = lda(train_X_norm, train_y, test_X_norm)
train_X_lpca, test_X_lpca = lda(train_X_pca1, train_y, test_X_pca1)
# train_X_norm_lda, test_X_norm_lda = lda(train_X_norm_pca, train_y, test_X_norm_pca)
dispSize(train_X_lda, train_y, test_X_lda, test_y, "Normalized LDA Dataset")
dispSize(train_X_lpca, train_y, test_X_lpca, test_y, "Normalized PCA LDA Dataset")


# ### Logistic Regresssion

# In[ ]:


print("Logistic Regression params")
log_reg = LogisticRegression()


# In[ ]:


# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 5)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
print("Logistic params: {}".format(hyperparameters))


# In[ ]:


# Create grid search using 5-fold cross validation
clf = GridSearchCV(log_reg, hyperparameters, cv=3, verbose=0)


# In[ ]:


print("--"*50)
print("Hyperparams on PCA")
best_model = clf.fit(train_X_pca, train_y)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
train_score = best_model.score(train_X_pca, train_y)
val_score = best_model.score(test_X_pca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LDA")
# Create grid search using 5-fold cross validation
clf = GridSearchCV(log_reg, hyperparameters, cv=3, verbose=0)
best_model = clf.fit(train_X_lda, train_y)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
train_score = best_model.score(train_X_lda, train_y)
val_score = best_model.score(test_X_lda, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LPCA")
# Create grid search using 5-fold cross validation
clf = GridSearchCV(log_reg, hyperparameters, cv=3, verbose=0)
best_model = clf.fit(train_X_lpca, train_y)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
train_score = best_model.score(train_X_lpca, train_y)
val_score = best_model.score(test_X_lpca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# ### Linear SVM

# In[ ]:


print("Linear SVMS params")
print("--"*50)
print("Hyperparams on PCA")
svm_clf = svm.SVC()
parameters = {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 100, 1000]}
clf = GridSearchCV(estimator=svm_clf, param_grid=parameters, cv=3, verbose=0)
best_model = clf.fit(train_X_pca, train_y)
# View best hyperparameters
print('Best C:', best_model.best_estimator_.get_params()['C'])
train_score = best_model.score(train_X_pca, train_y)
val_score = best_model.score(test_X_pca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LDA")
svm_clf = svm.SVC()
parameters = {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 100, 1000]}
clf = GridSearchCV(estimator=svm_clf, param_grid=parameters, cv=3, verbose=0)
best_model = clf.fit(train_X_lda, train_y)
# View best hyperparameters
print('Best C:', best_model.best_estimator_.get_params()['C'])
train_score = best_model.score(train_X_lda, train_y)
val_score = best_model.score(test_X_lda, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LPCA")
svm_clf = svm.SVC()
parameters = {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 100, 1000]}
clf = GridSearchCV(estimator=svm_clf, param_grid=parameters, cv=3, verbose=0)
best_model = clf.fit(train_X_lpca, train_y)
# View best hyperparameters
print('Best C:', best_model.best_estimator_.get_params()['C'])
train_score = best_model.score(train_X_lpca, train_y)
val_score = best_model.score(test_X_lpca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# ### SVM RBF Kernel

# In[ ]:


print("RBF Kernel params")

gamma_range = np.outer(np.logspace(-3, 0, 1),np.array([1,5]))
gamma_range = gamma_range.flatten()

# generate matrix with all C
#C_range = np.outer(np.logspace(-3, 3, 7),np.array([1,2, 5]))
C_range = np.outer(np.logspace(-1, 1, 2),np.array([1,5]))
# flatten matrix, change to 1D numpy array
C_range = C_range.flatten()

parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}
print("Hyperparams for RBF : {}".format(parameters))


# In[ ]:


print("--"*50)
print("Hyperparams on PCA")
svm_clsf = svm.SVC()
clf = GridSearchCV(estimator = svm_clsf,param_grid=parameters,n_jobs = 4, verbose=0)
best_model = clf.fit(train_X_pca, train_y)
# View best hyperparameters
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Gamma:', best_model.best_estimator_.get_params()['gamma'])
train_score = best_model.score(train_X_pca, train_y)
val_score = best_model.score(test_X_pca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LDA")
svm_clsf = svm.SVC()
clf = GridSearchCV(estimator = svm_clsf,param_grid=parameters,n_jobs = 4, verbose=0)
best_model = clf.fit(train_X_lda, train_y)
# View best hyperparameters
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Gamma:', best_model.best_estimator_.get_params()['gamma'])
train_score = best_model.score(train_X_lda, train_y)
val_score = best_model.score(test_X_lda, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LPCA")
svm_clsf = svm.SVC()
clf = GridSearchCV(estimator = svm_clsf,param_grid=parameters,n_jobs = 4, verbose=0)
best_model = clf.fit(train_X_lpca, train_y)
# View best hyperparameters
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Gamma:', best_model.best_estimator_.get_params()['gamma'])
train_score = best_model.score(train_X_lpca, train_y)
val_score = best_model.score(test_X_lpca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# ### Decision Tree Classifier

# In[ ]:


print("Decision Tree params : ")
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
}
print(random_grid)


# In[ ]:


print("--"*50)
print("Hyperparams on PCA")
dec_clf = DecisionTreeClassifier()
clf = GridSearchCV(dec_clf, random_grid, cv=5, verbose=0)
best_model = clf.fit(train_X_pca, train_y)
# View best hyperparameters
print(best_model.best_estimator_.get_params()['max_features'])
print(best_model.best_estimator_.get_params()['max_depth'])
print(best_model.best_estimator_.get_params()['min_sample_split'])
print(best_model.best_estimator_.get_params()['min_samples_leaf'])
train_score = best_model.score(train_X_pca, train_y)
val_score = best_model.score(test_X_pca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LDA")
dec_clf = DecisionTreeClassifier()
clf = GridSearchCV(dec_clf, random_grid, cv=5, verbose=0)
best_model = clf.fit(train_X_lda, train_y)
# View best hyperparameters
print(best_model.best_estimator_.get_params()['max_features'])
print(best_model.best_estimator_.get_params()['max_depth'])
print(best_model.best_estimator_.get_params()['min_sample_split'])
print(best_model.best_estimator_.get_params()['min_samples_leaf'])
train_score = best_model.score(train_X_lda, train_y)
val_score = best_model.score(test_X_lda, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)


# In[ ]:


print("--"*50)
print("Hyperparams on LPCA")
dec_clf = DecisionTreeClassifier()
clf = GridSearchCV(dec_clf, random_grid, cv=5, verbose=0)
best_model = clf.fit(train_X_lpca, train_y)
# View best hyperparameters
print(best_model.best_estimator_.get_params()['max_features'])
print(best_model.best_estimator_.get_params()['max_depth'])
print(best_model.best_estimator_.get_params()['min_sample_split'])
print(best_model.best_estimator_.get_params()['min_samples_leaf'])
train_score = best_model.score(train_X_lpca, train_y)
val_score = best_model.score(test_X_lpca, test_y)
print("Train Score: ", train_score)
print("Val Score: ", val_score)

