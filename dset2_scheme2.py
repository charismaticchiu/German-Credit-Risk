"""
Author: Ming-Chang Chiu
Date: 29th April 2017
Acknowledgement: 
  All functions used is from Scikit-learn Library
  The organization of the code and Pipeline modified from Scikit-learn example,
  "Sample pipeline for text feature extraction and evaluation(http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py)"

  Handling of arrays credit to numpy Library
"""
import csv
import numpy as np
import scipy 
import sklearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from numpy.random import randint
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def processCSV(filename):
  data = pd.read_csv(filename, header = None)

  checking    = []
  duration    = np.asarray(data[1],dtype = np.float32)
  credit_his  = [] 
  
  credit      = np.asarray(data[4],dtype = np.float32)
  saving      = []
  employ      = []
  inst        = np.asarray(data[7],dtype = np.float32)
  status_sex  = []
  other_status = []
  residence_since = np.asarray(data[10],dtype = np.float32)
  prop        = [] # Peoperty
  age         = np.asarray(data[12],dtype = np.float32)
  other_insl  = []
  housing     = []
  num_credit  = np.asarray(data[15],dtype = np.float32)
  job         = []
  num_liable  = np.asarray(data[17],dtype = np.float32)
  tel         = []
  foreign     = []

  label       = np.asarray(data[20],dtype = np.float32)



  
  
  for i in xrange(1000):
      
    if data[0][i] == 'A11':
      checking.append(-1.)
    elif data[0][i] == 'A12':
      checking.append(1.)
    elif data[0][i] == 'A13':
      checking.append(2.)
    elif data[0][i] == 'A14':
      checking.append(0.)
    
    if   data[2][i] == 'A30':
      credit_his.append(1.)
    elif data[2][i] == 'A31':
      credit_his.append(1.)
    elif data[2][i] == 'A32':
      credit_his.append(1.)
    elif data[2][i] == 'A33':
      credit_his.append(-1.)
    elif data[2][i] == 'A34':
      credit_his.append(1.)

    """
    if   data[3][i] == 'A40':
      purpose.append(-1.)
    elif data[3][i] == 'A41':
      purpose.append(-1.)
    elif data[3][i] == 'A42':
      purpose.append(-1.)
    elif data[3][i] == 'A43':
      purpose.append(-1.)
    elif data[3][i] == 'A44':
      purpose.append(-1.)
    elif data[3][i] == 'A45':
      purpose.append(-1.)
    elif data[3][i] == 'A46':
      purpose.append(-1.)
    elif data[3][i] == 'A47':
      purpose.append(-1.)
    elif data[3][i] == 'A48':
      purpose.append(-1.)
    elif data[3][i] == 'A49':
      purpose.append(-1.)
    elif data[3][i] == 'A410':
      purpose.append(-1.)
    """

    if   data[5][i] == 'A61':
      saving.append(1.)
    elif data[5][i] == 'A62':
      saving.append(2.)
    elif data[5][i] == 'A63':
      saving.append(3.)
    elif data[5][i] == 'A64':
      saving.append(4.)
    elif data[5][i] == 'A65':
      saving.append(0.)

    if   data[6][i] == 'A71':
      employ.append(-1.)
    elif data[6][i] == 'A72':
      employ.append(0.)
    elif data[6][i] == 'A73':
      employ.append(1.)
    elif data[6][i] == 'A74':
      employ.append(2.)
    elif data[6][i] == 'A75':
      employ.append(3.)

    if   data[8][i] == 'A91':
      status_sex.append([1.,0.,0.,0.,0.])
    elif data[8][i] == 'A92':
      status_sex.append([0.,1.,0.,0.,0.])
    elif data[8][i] == 'A93':
      status_sex.append([0.,0.,1.,0.,0.])
    elif data[8][i] == 'A94':
      status_sex.append([0.,0.,0.,1.,0.])
    elif data[8][i] == 'A95':
      status_sex.append([0.,0.,0.,0.,1.])

    if   data[9][i] == 'A101':
      other_status.append(0)
    elif data[9][i] == 'A102':
      other_status.append(1.)
    elif data[9][i] == 'A103':
      other_status.append(1.)

    if   data[11][i] == 'A121':
      prop.append(3.)
    elif data[11][i] == 'A122':
      prop.append(2.)
    elif data[11][i] == 'A123':
      prop.append(1.)
    elif data[11][i] == 'A124':
      prop.append(0.)
    
    if   data[13][i] == 'A141':
      other_insl.append(1.)
    elif data[13][i] == 'A142':
      other_insl.append(1.)
    elif data[13][i] == 'A143':
      other_insl.append(0.)

    if   data[14][i] == 'A151':
      housing.append(0.)
    elif data[14][i] == 'A152':
      housing.append(2.)
    elif data[14][i] == 'A153':
      housing.append(1.)
    
    if   data[16][i] == 'A171':
      job.append(-1.)
    elif data[16][i] == 'A172':
      job.append(0.)
    elif data[16][i] == 'A173':
      job.append(1.)
    elif data[16][i] == 'A174':
      job.append(2.)

    if   data[18][i] == 'A191':
      tel.append(0.)
    elif data[18][i] == 'A192':
      tel.append(1.)

    if   data[19][i] == 'A201':
      foreign.append([1., 0.])
    elif data[19][i] == 'A202':
      foreign.append([0., 1.])
    
  return np.asarray(checking).reshape(-1,1),duration.reshape(-1,1),np.asarray(credit_his).reshape(-1,1),credit.reshape(-1,1),np.asarray(saving).reshape(-1,1),np.asarray(employ).reshape(-1,1),inst.reshape(-1,1),np.asarray(status_sex),  np.asarray(other_status).reshape(-1,1),residence_since.reshape(-1,1),np.asarray(prop).reshape(-1,1),age.reshape(-1,1),np.asarray(other_insl).reshape(-1,1),np.asarray(housing).reshape(-1,1),num_credit.reshape(-1,1),np.asarray(job).reshape(-1,1),num_liable.reshape(-1,1),np.asarray(tel).reshape(-1,1),np.asarray(foreign),label.reshape(-1,1)       
  

if __name__ == '__main__':

  filename1 = 'Proj_dataset_2.csv'
  checking ,duration , credit_his ,credit , saving , employ ,inst , status_sex , other_status ,residence_since , prop ,age , other_insl, housing,num_credit , job ,num_liable , tel , foreign ,label = processCSV(filename1)
  
  data = np.hstack((checking ,duration , credit_his ,credit , saving , employ ,inst , status_sex , other_status ,residence_since , prop ,age , other_insl, housing,num_credit , job ,num_liable , tel , foreign))
  
  names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
          "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"]
  classifiers = [
    KNeighborsClassifier(20),
    SVC(kernel="linear", C=0.5),
    SVC(gamma=2, C=2),
    
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1.),
    MLPClassifier(solver='lbfgs',max_iter=10000,alpha=1e-5, hidden_layer_sizes=(17,10,3,1), random_state=1,early_stopping=True),
    AdaBoostClassifier(),
    GaussianNB(),
    
  ]
  param_grids = [
    {'sel__n_components': xrange(5,data.shape[1]+1), "clf__n_neighbors": [15,20,25]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__kernel': ['linear'], 'clf__C': [0.5, 1, 3, 5]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__kernel': ['rbf'], 'clf__gamma': [1, 2,1e-1, 1e-2], 'clf__C': [1, 2, 5]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__max_depth':[1,3,5], 'clf__n_estimators':[10,20,40,45],'clf__max_features':[0.25,0.5,1.]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__alpha':[1e-5,1e-6],'clf__hidden_layer_sizes':[(17,10,3,1), (17,12,1), 
                                                                                            (16,10,3,1), (16,12,1), 
                                                                                            (15,10,3,1), (15,10,1), 
                                                                                            (14,10,3,1), (14, 7,1), 
                                                                                            (13,10,3,1), (13, 7,1), 
                                                                                            (12,10,3,1), (12, 6,1), 
                                                                                            (11,10,3,1), (11, 6,1), 
                                                                                            (10,10,3,1), (10, 5,1), 
                                                                                            ( 9, 6,3,1), ( 9, 5,1), 
                                                                                            ( 8, 5,3,1), ( 8, 4,1), 
                                                                                            ]},
    {'sel__n_components': xrange(5,data.shape[1]+1)},
    {'sel__n_components': xrange(5,data.shape[1]+1)},
    
  ]
  param_grids_2 = [
    {'sel__n_components': xrange(5,data.shape[1]+1), "clf__n_neighbors": [15,20,25]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__kernel': ['linear'], 'clf__C': [0.5, 1, 3, 5]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__kernel': ['rbf'], 'clf__gamma': [1, 2,1e-1, 1e-2], 'clf__C': [1, 2, 5]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__max_depth':[1,3,5], 'clf__n_estimators':[10,20,40,45],'clf__max_features':[0.25,0.5,1.]},
    {'sel__n_components': xrange(5,data.shape[1]+1), 'clf__alpha':[1e-5,1e-6],'clf__hidden_layer_sizes':[(17,10,3,1), (17,3,1),
                                                                                            (16,10,3,1), (16,3,1),
                                                                                            (15,10,3,1), (15,3,1),
                                                                                            (14,10,3,1), (14,3,1),
                                                                                            (13,10,3,1), (13,3,1),
                                                                                            (12,10,3,1), (12,3,1),
                                                                                            (11,10,3,1), (11,3,1),
                                                                                            (10,10,3,1), (10,3,1),
                                                                                            ( 9, 6,3,1), ( 9,3,1),
                                                                                            ( 8, 5,3,1), ( 8,3,1),
                                                                                            ]},
    {'sel__n_components': xrange(5,data.shape[1]+1)},
    {'sel__n_components': xrange(5,data.shape[1]+1)},
    
  ]
  pipelines = [
    Pipeline([('sel', PCA()), ('clf', KNeighborsClassifier(20))]),
    Pipeline([('sel', PCA()), ('clf', SVC(kernel="linear", C=0.5))]),
    Pipeline([('sel', PCA()), ('clf', SVC(gamma=2, C=2))]),
    Pipeline([('sel', PCA()), ('clf', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1.))]),
    Pipeline([('sel', PCA()), ('clf', MLPClassifier())]),
    Pipeline([('sel', PCA()), ('clf', AdaBoostClassifier())]),
    Pipeline([('sel', PCA()), ('clf', GaussianNB())]),
    
  ]

  pipelines_2 = [
    Pipeline([('sel', LinearDiscriminantAnalysis()), ('clf', KNeighborsClassifier(20))]),
    Pipeline([('sel', LinearDiscriminantAnalysis()), ('clf', SVC(kernel="linear", C=0.5))]),
    Pipeline([('sel', LinearDiscriminantAnalysis()), ('clf', SVC(gamma=2, C=2))]),
    Pipeline([('sel', LinearDiscriminantAnalysis()), ('clf', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1.))]),
    Pipeline([('sel', LinearDiscriminantAnalysis()), ('clf', MLPClassifier())]),
    Pipeline([('sel', LinearDiscriminantAnalysis()), ('clf', AdaBoostClassifier())]),
    Pipeline([('sel', LinearDiscriminantAnalysis()), ('clf', GaussianNB())]),
    
  ]

  
  label = label.reshape(1000,)
  X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

  scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
  #print scaler.scale_
  #print scaler.mean_
  #print scaler.var_
  X_train_transformed = scaler.transform(X_train)
  X_test_transformed = scaler.transform(X_test)
  
  u, c = np.unique(y_train, return_counts=True)
  c = c.astype(np.float32)
  print 'Prior (Baseline):', c[0]/(c[0]+c[1])

  

  for name, pipeline, param_grid, pipeline_2, param_grid_2 in zip(names, pipelines, param_grids, pipelines_2, param_grids_2):


      print 
      print '------',name,'------'
      print 
      
      clf_post = GridSearchCV(pipeline, param_grid, cv=4)
      clf_post.fit(X_train_transformed, y_train) #######
      print clf_post.best_params_
      print clf_post.best_score_
      #print clf_post.cv_results_
      print
      #X_test_new = selector.transform(X_test_transformed)
      y_pred = clf_post.predict(X_test_transformed)
      print classification_report(y_test, y_pred)
      print 'Test Accuracy: ',np.mean(y_test == y_pred)

      ### LDA
      print
      print "---- LDA ----"
      print
      clf_post = GridSearchCV(pipeline_2, param_grid_2, cv=4)
      clf_post.fit(X_train_transformed, y_train) #######
      print clf_post.best_params_
      print clf_post.best_score_
      #print clf_post.cv_results_
      print
      #X_test_new = selector.transform(X_test_transformed)
      y_pred = clf_post.predict(X_test_transformed)
      print classification_report(y_test, y_pred)
      print 'Test Accuracy: ',np.mean(y_test == y_pred)
  