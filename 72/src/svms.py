
# svms

import numpy as np
import pandas as pd
from scipy import io
from skmultilearn import problem_transform
import skmultilearn.model_selection
from sklearn import model_selection, svm, preprocessing, ensemble, naive_bayes, neighbors, tree, metrics
import matplotlib.pyplot as plt
import matplotlib

import data
import functions as f
import settings

import time, datetime
start = time.time()


TODROP = ['split\d', 'time', 'accuracy', 'f1', 'std']
TODROP2 = ['split\d', 'time', 'std']


INFOFILE = '../logs/svms.csv'
f.resetFile(INFOFILE)


x = data.x
y = data.y
k = data.k
l = data.l


models = {}
results = {}
allcv = pd.DataFrame()


# parameters svms
parameters = [{
  'classifier': [svm.SVC()],
  'classifier__C': np.logspace(-4, 4, 9),
  'classifier__gamma' :  np.logspace(-4, 4, 9),
}]

metrics_y = ['accuracy', 'f1_micro', 'hamming_loss']

refit = 'f1_micro'

# binary relevance - svms


model = f.training_phase(problem_transform.BinaryRelevance(), parameters, x, y, metrics_y, refit)
test_result = f.testing_phase(model, k, l, metrics_y)
test_result['model'] = 'binary_relevance_svms'
test_results = test_result


models['br_svc'] = model
result = model.cv_results_
results['br_svc'] = result
result = pd.DataFrame(result)
f.drop_write(result, TODROP2, INFOFILE)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'binary_relevance_svms'
best_results = best_result
allcv = allcv.append(best_result)


# classifier chains - svms


model = f.training_phase(problem_transform.ClassifierChain(), parameters, x, y, metrics_y, refit)
test_result = f.testing_phase(model, k, l, metrics_y)
test_result['model'] = 'classifier_chain_svms'
test_results = test_results.append(test_result)


models['br_svc'] = model
result = model.cv_results_
results['br_svc'] = result
result = pd.DataFrame(result)
f.drop_write(result, TODROP2, INFOFILE)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'classifier_chain_svms'
best_results = best_results.append(best_result)
allcv = allcv.append(best_result)


# label powerset - svms


model = f.training_phase(problem_transform.LabelPowerset(), parameters, x, y, metrics_y, refit)
test_result = f.testing_phase(model, k, l, metrics_y)
test_result['model'] = 'label_powerset_svms'
test_results = test_results.append(test_result)


models['br_svc'] = model
result = model.cv_results_
results['br_svc'] = result
result = pd.DataFrame(result)
f.drop_write(result, TODROP2, INFOFILE)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'label_powerset_svms'
best_results = best_results.append(best_result)
allcv = allcv.append(best_result)


# plot final scores for all models
print(test_results)
f.drop_write(test_results, TODROP2, INFOFILE)
f.plot_and_save(
  data=f.metrics_test(test_results, metrics_y=metrics_y),
  filename='../plots/svms/all_models_all_final_scores.png',
  x='model',
  title='best final scores on test set - all models - svm base',
  ylabel='score'
)


# plot cv test performance for all models
print(allcv)
f.drop_write(allcv, TODROP2, INFOFILE)
f.plot_and_save(
  data=f.metrics_cv(allcv, set='test', x='model', metrics_y=metrics_y),
  filename='../plots/svms/all_models_all_cv_test_scores.png',
  x='model',
  title='best cross validation performance - all models - svm base - test',
  ylabel='score'
)

# plot cv train performance for all models
print(allcv)
f.drop_write(allcv, TODROP2, INFOFILE)
f.plot_and_save(
  data=f.metrics_cv(allcv, set='train', x='model', metrics_y=metrics_y),
  filename='../plots/svms/all_models_all_cv_train_scores.png',
  x='model',
  title='best cross validation performance - all models - svm base - train',
  ylabel='score'
)
print('run time, ', str(datetime.timedelta(seconds=(time.time() - start))), ', seconds')
