
# functions

import numpy as np
import pandas as pd
from skmultilearn import problem_transform
from sklearn import model_selection, metrics, svm
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def scorer_avg_precision(estimator, X, y):
  return metrics.average_precision_score(y, estimator.predict_proba(X).toarray(), average='micro')

def scorer_roc_auc(estimator, X, y):
  return metrics.roc_auc_score(y, estimator.predict_proba(X).toarray(), average='micro')

def scorer_coverage_error(estimator, X, y):
  return metrics.coverage_error(y, estimator.predict_proba(X).toarray())

def scorer_lrap(estimator, X, y):
  return metrics.label_ranking_average_precision_score(y, estimator.predict_proba(X).toarray())  

def scorer_label_ranking_loss(estimator, X, y):
  return -metrics.label_ranking_loss(y, estimator.predict_proba(X).toarray())  


def metrics_cv(data, set='test', metrics_y=[], x=None):
  d = {}
  if x is not None:
    d[x] = list(data[x])
  for metric in metrics_y:
    d[metric] = list(data['mean_'+str(set)+'_'+str(metric)])
  return pd.DataFrame(d)  

def metrics_test(data, metrics_y=[]):
  d = {}
  d['model'] = list(data['model'])
  for i in metrics_y:
    d[i] = list(data[i])
  return pd.DataFrame(d)


def training_phase(model, params, x, y, metrics_y, refit):
    scoring = {}
    for i in metrics_y:

      if i is 'accuracy':
        scoring[i] = i
        
      elif i is 'f1_micro':
        scoring[i] = i

      elif i is 'hamming_loss':
        scoring[i] = metrics.make_scorer(metrics.hamming_loss, greater_is_better=False)

      elif i is 'precision_micro':
        scoring[i] = i

      elif i is 'recall_micro':
        scoring[i] = i

      elif i is 'roc_auc':
        scoring[i] = scorer_roc_auc

      elif i is 'average_precision':
        scoring[i] = scorer_avg_precision

      elif i is 'coverage_error':
        scoring[i] = scorer_coverage_error

      elif i is 'label_ranking_average_precision_score':
        scoring[i] = scorer_lrap

      elif i is 'label_ranking_loss':
        scoring[i] = scorer_label_ranking_loss

    model = model_selection.GridSearchCV(
      model,
      params,
      scoring=scoring,
      refit=refit,
      return_train_score=True,
      cv=10,
      n_jobs=10,
      verbose=1
    )
    model.fit(x, y)
    return model


def testing_phase(model, k, l, metrics_y):
    l_star = model.predict(k)
    result = {}
    for i in metrics_y:

      if i is 'accuracy':
        result[i] = [metrics.accuracy_score(l, l_star)]
        
      elif i is 'f1_micro':
        result[i] = [metrics.f1_score(l, l_star, average='micro')]

      elif i is 'hamming_loss':
        result[i] = [-metrics.hamming_loss(l, l_star)]

      elif i is 'precision_micro':
        result[i] = [metrics.precision_score(l, l_star, average='micro')]

      elif i is 'recall_micro':
        result[i] = [metrics.recall_score(l, l_star, average='micro')]

      elif i is 'roc_auc':
        result[i] = [scorer_roc_auc(model, k, l)]

      elif i is 'average_precision':
        result[i] = [scorer_avg_precision(model, k, l)]

      elif i is 'coverage_error':
        result[i] = [scorer_coverage_error(model, k, l)]

      elif i is 'label_ranking_average_precision_score':
        result[i] = [scorer_lrap(model, k, l)]

      elif i is 'label_ranking_loss':
        result[i] = [scorer_label_ranking_loss(model, k, l)]

    return pd.DataFrame(result)


def get_best_result(model):
    return pd.DataFrame(model.cv_results_).iloc[[model.best_index_]]


def plot_and_save(data, filename, **kwargs):
  data.plot.bar(rot=15, **kwargs)
  plt.legend(loc='lower left')
  plt.grid()
  plt.savefig(filename)
  return


def getTableAsCsv(x):
  '''
  convert data to csv format
  '''
  return pd.DataFrame(x).round(4).to_csv()


def writeTable(x, name):
  '''
  write data in csv format to a file
  '''
  file = open(str(name), 'a')
  file.write(getTableAsCsv(x) + '\n')
  file.close()
  return

def dropColumns(x, regexs):
  '''
  drop columns from a dataframe base on regular expresions
  '''
  y = x.copy()
  for regex in regexs:
    y.drop(y.filter(regex=regex).columns, axis = 1, inplace = True)
  return y

def drop_write(dataframe, regexs, filename):
  writeTable(dropColumns(dataframe, regexs), filename)
  return

def resetFile(x):
  file = open(x, 'w')
  file.close()
  return
