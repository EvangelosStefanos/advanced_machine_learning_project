import numpy as np
import pandas as pd
from scipy import io
from skmultilearn.problem_transform import BinaryRelevance
from sklearn import model_selection, svm, preprocessing, \
  ensemble, naive_bayes, neighbors, tree
import matplotlib.pyplot as plt

data = io.loadmat('../data/CHD_49.mat')
x = data['data']
y = data['targets']

x = preprocessing.MinMaxScaler().fit_transform(x)
y = preprocessing.MinMaxScaler().fit_transform(y)



parameters = [{
  'classifier': [
    ensemble.RandomForestClassifier(),
    naive_bayes.MultinomialNB(),
    neighbors.KNeighborsClassifier(),
    svm.LinearSVC(),
    svm.SVC(),
    tree.DecisionTreeClassifier()
  ]
},]

{
  'classifier': [svm.SVC()],
  'classifier__C': np.logspace(-4, 4, 9),
  'classifier__gamma' :  np.logspace(-4, 4, 9)
}

model = model_selection.GridSearchCV(
  BinaryRelevance(),
  parameters,
  scoring=['accuracy', 'f1_macro'],
  refit='f1_macro',
  return_train_score=True
)


model.fit(x, y)

print (model.best_params_, model.best_score_)

d = pd.DataFrame(model.cv_results_)

def dropColumns(x, regexs):
  '''
  drop columns from a dataframe base on regular expresions
  '''
  y = x.copy()
  for regex in regexs:
    y.drop(y.filter(regex=regex).columns, axis = 1, inplace = True)
  return y

def sortTable(x, sortby=None):
  '''
  sort a dataframe based on some column
  '''
  if(sortby is not None):
    x = x.sort_values(sortby, ascending=False)
  return x

d = sortTable(d, 'rank_test_f1_macro')
d = dropColumns(d, ['std', 'split\d', 'time', 'params', 'rank', 'f1_macro', 'train'])

d.plot.bar(x='param_classifier')
plt.xticks(rotation=10)
plt.show()
print(d)

