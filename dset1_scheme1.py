"""
Author: Ming-Chang Chiu
Date: 29th April 2017
Acknowledgement: 
  All functions credits to Scikit-learn Library
  The organization of the code and Pipeline usage are modified from Scikit-learn example,
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
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def processCSV(filename):
  """
  Credits: read_csv function is from Pandas library
  """
  data = pd.read_csv(filename)

  age = np.asarray(data.Age,dtype = np.float32)
  sex = []
  job = np.asarray(data.Job,dtype = np.float32)
  credit = np.asarray(data['Credit amount'])
  saving = []
  checking = []
  duration = np.asarray(data.Duration,dtype = np.float32)
  housing = []
  purpose = []
  label = np.asarray(data.Class,dtype = np.float32)

  
  for i in xrange(1000):
      
    if data.Sex[i] == 'male':
      sex.append([1.,0.])
    else: sex.append([0.,1.])

    

    if data.Housing[i] == 'own': # can be unordered
      housing.append(2.)
    elif data.Housing[i] == 'rent':
      housing.append(1.)
    else: housing.append(0.)

    if data['Saving accounts'][i] == 'NA': # can be unordered
      saving.append(0.)
    elif data['Saving accounts'][i] == 'little':
      saving.append(1.)
    elif data['Saving accounts'][i] == 'moderate':
      saving.append(2.)
    elif data['Saving accounts'][i] == 'quite rich':
      saving.append(3.)
    else: 
      saving.append(4.)

    if data['Checking account'][i] == 'NA': # can be unordered
      checking.append(0.)
    elif data['Checking account'][i] == 'little':
      checking.append(1.)
    elif data['Checking account'][i] == 'moderate':
      checking.append(2.)
    elif data['Checking account'][i] == 'quite rich':
      checking.append(3.)
    else: 
      checking.append(4.)


    if data['Purpose'][i] == 'car': # can be unordered
      purpose.append([1.,0.,0.,0.,0.,0.,0.,0.])
    elif data['Purpose'][i] == 'furniture/equipment':
      purpose.append([0.,1.,0.,0.,0.,0.,0.,0.])
    elif data['Purpose'][i] == 'radio/TV':
      purpose.append([0.,0.,1.,0.,0.,0.,0.,0.])
    elif data['Purpose'][i] == 'domestic appliances':
      purpose.append([0.,0.,0.,1.,0.,0.,0.,0.])
    elif data['Purpose'][i] == 'repairs':
      purpose.append([0.,0.,0.,0.,1.,0.,0.,0.])
    elif data['Purpose'][i] == 'education':
      purpose.append([0.,0.,0.,0.,0.,1.,0.,0.])
    elif data['Purpose'][i] == 'business':
      purpose.append([0.,0.,0.,0.,0.,0.,1.,0.])
    else: purpose.append([0.,0.,0.,0.,0.,0.,0.,1.])

    

  return age.reshape(-1,1),np.asarray(sex),job.reshape(-1,1),np.asarray(housing).reshape(-1,1),np.asarray(saving).reshape(-1,1),np.asarray(checking).reshape(-1,1),credit.reshape(-1,1),duration.reshape(-1,1),np.asarray(purpose),label.reshape(-1,1)

if __name__ == '__main__':
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
    {"n_neighbors": [15,20,25]},
    {'kernel': ['linear'], 'C': [0.5, 1, 3, 5]},
    {'kernel': ['rbf'], 'gamma': [1, 2,1e-1, 1e-2], 'C': [1, 2, 5]},
    
    {'max_depth':[1,3,5], 'n_estimators':[10,20,40,45],'max_features':[0.25,0.5,1.]},
    {'alpha':[1e-5,1e-6],'hidden_layer_sizes':[(17,10,3,1), (17,10,1), (17,3,1),(17,12,3,1)]},
    {},
    {},
    
  ]
  filename1 = 'Proj_dataset_1.csv'
  age,sex,job,housing,saving,checking,credit,duration,purpose,label = processCSV(filename1)
  """
  age = age/np.max(age)
  job = job/np.max(job)
  housing = housing/np.max(housing)
  saving = saving/np.max(saving)
  checking = checking/np.max(checking)
  credit = credit/np.max(credit)
  duration = duration/np.max(duration)
  """
  data = np.hstack((age,sex,job,housing,saving,checking,credit,duration,purpose))
  label = label.reshape(1000,)
  X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

  scaler = StandardScaler(with_mean = True, with_std = True).fit(X_train)
  #print scaler.scale_
  #print scaler.mean_
  #print scaler.var_
  X_train_transformed = scaler.transform(X_train)
  X_test_transformed = scaler.transform(X_test)

  u, c = np.unique(y_train, return_counts=True)
  c = c.astype(np.float32)
  print 'Prior (Baseline):', c[0]/(c[0]+c[1])

  
  

  for name, clf, i in zip(names, classifiers, xrange(7)):

    pca_score = []
    lda_score = []
    pca_model = []
    lda_model = []
    pca_param = []
    lda_param = []
    pca_selec = []
    lda_selec = []
    pca_k = []
    lda_k = []
    for k in xrange(5,18):

      #print 
      #print '------',name,k,'------'
      #print 
      if i == 4:
        param_grids[4] = {'alpha':[1e-5,1e-6],'hidden_layer_sizes':[(k,10,3,1), (k,12,3,1), (k,3,1)]}
      selector = PCA(k).fit(X_train_transformed,y_train)
      X_train_new = selector.transform(X_train_transformed)
    
      #clf.fit(X_train_transformed, y_train)
      clf_post = GridSearchCV(clf, param_grids[i], cv=4)
      clf_post.fit(X_train_new, y_train) #######

      pca_param.append(clf_post.best_params_)
      pca_score.append(clf_post.best_score_)
      pca_selec.append(selector)
      pca_model.append(clf_post)
      pca_k.append(k)
      
      """
      print clf_post.best_params_
      print clf_post.best_score_
      print 
      X_test_new = selector.transform(X_test_transformed)
      y_pred = clf_post.predict(X_test_new)
      
      print classification_report(y_test, y_pred)
      print 'Accuracy: ',np.mean(y_test == y_pred)
      """
      #print
      #print "---- LDA ----"
      #print
      selector = LinearDiscriminantAnalysis(n_components=k).fit(X_train_transformed,y_train)
      X_train_new = selector.transform(X_train_transformed)
    
      #clf.fit(X_train_transformed, y_train)
      clf_post = GridSearchCV(clf, param_grids[i], cv=4)
      clf_post.fit(X_train_new, y_train) #######

      lda_param.append(clf_post.best_params_)
      lda_score.append(clf_post.best_score_)
      lda_selec.append(selector)
      lda_model.append(clf_post)
      lda_k.append(k)
      """
      print clf_post.best_params_
      print clf_post.best_score_
      print 
      X_test_new = selector.transform(X_test_transformed)
      y_pred = clf_post.predict(X_test_new)
      
      print classification_report(y_test, y_pred)
      print 'Accuracy: ',np.mean(y_test == y_pred)
      
      """
    pca_idx = np.argmax(pca_score)
    lda_idx = np.argmax(lda_score)
    print
    print '------',name,'------'
    print
    X_test_new = pca_selec[pca_idx].transform(X_test_transformed)
    print 'dim',pca_k[pca_idx]
    print 'best score: ',pca_score[pca_idx]
    print 'best param: ',pca_param[pca_idx]
    y_pred = pca_model[pca_idx].predict(X_test_new)
    print classification_report(y_test, y_pred)
    print 'Test Accuracy: ',np.mean(y_test == y_pred)

    print 
    print "---- LDA ----"
    print 
    X_test_new = lda_selec[lda_idx].transform(X_test_transformed)
    print 'dim',lda_k[lda_idx]
    print 'best score: ',lda_score[lda_idx]
    print 'best param: ',lda_param[lda_idx]
    y_pred = lda_model[lda_idx].predict(X_test_new)
    print classification_report(y_test, y_pred)
    print 'Test Accuracy: ',np.mean(y_test == y_pred)
