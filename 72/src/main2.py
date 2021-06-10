#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy import io
from skmultilearn import problem_transform
import skmultilearn.model_selection
from sklearn import model_selection, svm, preprocessing,   ensemble, naive_bayes, neighbors, tree, metrics
import matplotlib.pyplot as plt


# In[3]:

# load data

data = io.loadmat('../data/CHD_49.mat')
x = data['data']
y = data['targets']

# transform labels to {0,1}

y = preprocessing.Binarizer().fit_transform(y)

# split data

x, y, k, l = skmultilearn.model_selection.iterative_train_test_split(x, y, test_size = 0.25)

# scale data

scaler = preprocessing.MinMaxScaler()
scaler = scaler.fit(x)
x = scaler.transform(x)
k = scaler.transform(k)


# In[4]:


parameters_6_classifiers = [{
  'classifier': [
    ensemble.RandomForestClassifier(),
    naive_bayes.MultinomialNB(),
    neighbors.KNeighborsClassifier(),
    svm.LinearSVC(),
    svm.SVC(),
    tree.DecisionTreeClassifier()
  ],
},]

parameters_svc = [{
  'classifier': [svm.SVC()],
  'classifier__C': np.logspace(-4, 4, 9),
  'classifier__gamma' :  np.logspace(-4, 4, 9)
}]


# In[60]:


def foo(model, params):
    model = model_selection.GridSearchCV(
      model,
      params,
      scoring={
          'accuracy': 'accuracy',
          'f1_macro': 'f1_macro',
          'f1_micro': 'f1_micro',
          'hamming_score': metrics.make_scorer(metrics.hamming_loss)
      },
      refit='accuracy',
      return_train_score=True
    )


    model.fit(x, y)
    print (model.best_params_, model.best_score_)
    print(pd.DataFrame(model.cv_results_))
    return model


# In[7]:

models = {}
results = {}

model = foo(problem_transform.BinaryRelevance(), parameters_6_classifiers)
models['br_6c'] = model

result = model.cv_results_
results['br_6c'] = result

result = pd.DataFrame(result)
# In[8]:


#plot(result, ['param_classifier'], ['mean_test_accuracy', 'mean_train_accuracy'])
result.plot.bar(x='param_classifier', y='mean_test_accuracy')
plt.rcParams['figure.figsize'] = [10, 4]
result.filter(regex='^mean_(test|train)_accuracy$|param_classifier').plot.bar(x='param_classifier', subplots=False, rot=15)


# In[9]:


model = foo(problem_transform.ClassifierChain(), parameters_6_classifiers)
models['cc_6c'] = model

result = model.cv_results_
results ['cc_6c']= result

# In[10]:


result.filter(regex='^mean_(test|train)_accuracy$|param_classifier').plot.bar(x='param_classifier', subplots=False, rot=15)
result.filter(regex='^mean_(test|train)_f1_macro$|param_classifier').plot.bar(x='param_classifier', subplots=False, rot=15)
result.filter(regex='^mean_(test|train)_f1_micro$|param_classifier').plot.bar(x='param_classifier', subplots=False, rot=15)
result.filter(regex='^mean_(test|train)_hamming_score$|param_classifier').plot.bar(x='param_classifier', subplots=False, rot=15)


# In[30]:


model = foo(problem_transform.LabelPowerset(), parameters_6_classifiers)
models['lp_6c'] = model

result = model.cv_results_
results['lp_6c'] = result

# In[61]:


model = foo(problem_transform.BinaryRelevance(), parameters_svc)
models['br_svc'] = model

# In[68]:


result = model.cv_results_
results['br_svc'] = result

result = pd.DataFrame(result)

best_column = d.iloc[model.best_index_]
print(best_column)


# In[80]:


data = {
    'classifier': ['svm'],
    'accuracy': best_column['mean_test_accuracy']
}
pd.DataFrame(data).plot.bar(x='classifier', rot=15)


# In[81]:


data = {
    'classifier': ['svm'],
    'f1_macro': best_column['mean_test_f1_macro']
}
pd.DataFrame(data).plot.bar(x='classifier', rot=15)


# In[82]:


d = pd.DataFrame()
d['param_classifier'] = results['br_6c']['param_classifier']
d['mean_test_accuracy_br'] = results['br_6c']['mean_test_accuracy']
d['mean_test_accuracy_cc'] = results['cc_6c']['mean_test_accuracy']
d['mean_test_accuracy_lp'] = results['lp_6c']['mean_test_accuracy']
d.plot.bar(x='param_classifier', rot=15)


# In[ ]:




