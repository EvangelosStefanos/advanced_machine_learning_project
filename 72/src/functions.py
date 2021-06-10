
# functions

import numpy as np
import pandas as pd
from skmultilearn import problem_transform
from sklearn import model_selection, metrics
import matplotlib.pyplot as plt
import matplotlib


metric_names = ['accuracy', 'f1_micro', 'hamming_loss']


def training_phase(model, params, x, y):
    model = model_selection.GridSearchCV(
      model,
      params,
      scoring={
          'accuracy': 'accuracy',
          'f1_micro': 'f1_micro',
          'hamming_loss': metrics.make_scorer(metrics.hamming_loss, greater_is_better=False),
          'precision': metrics.make_scorer(metrics.precision_score, average='micro'),
      },
      refit='hamming_loss',
      return_train_score=True,
      cv=10,
      n_jobs=10,
      verbose=1
    )
    model.fit(x, y)
    return model


def testing_phase(model, k, l):
    l_star = model.predict(k)
    result = {}
    result['accuracy'] = [metrics.accuracy_score(l, l_star)]
    result['f1_micro'] = [metrics.f1_score(l, l_star, average='micro')]
    result['hamming_loss'] = [-metrics.hamming_loss(l, l_star)]
    result['precision'] = [metrics.precision_score(l, l_star, average='micro')]    
    return pd.DataFrame(result)


def get_best_result(model):
    return pd.DataFrame(model.cv_results_).iloc[[model.best_index_]]


def get_metrics(result):
    res = {}
    for i in metric_names:
        res[i] = [result['mean_test_'+str(i)]]
    return res


def plot_and_save(data, filename, **kwargs):
  data.plot.bar(rot=15, **kwargs)
  plt.grid()
  plt.savefig(filename)
  return


def hard_plot(y, xlabels, ylabel, labels, nsamples, ngroups):
  x = np.arange(nsamples)
  width = .4

  fig, ax = plt.subplots()
  x_i = x - width/ngroups
  for y_i, l_i in zip(y, labels):
    ax.bar(x_i, y_i, width, label=l_i)
    x_i += width
  ax.set_xticks(x)
  ax.set_xticklabels(xlabels)
  ax.set_ylabel(ylabel)
  ax.legend()
  return ax


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
