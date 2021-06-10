
# random forests

import numpy as np
import pandas as pd
from scipy import io
from skmultilearn import problem_transform
import skmultilearn.model_selection
from sklearn import model_selection, svm, preprocessing, ensemble,\
  naive_bayes, neighbors, tree, metrics
import matplotlib.pyplot as plt
import matplotlib

import data
import functions as f
import settings


TODROP = ['split\d', 'time', 'accuracy', 'f1', 'std']
TODROP2 = ['split\d', 'time', 'std']


INFOFILE = '../random_forest_base/info.csv'
f.resetFile(INFOFILE)

x = data.x
y = data.y
k = data.k
l = data.l


models = {}
results = {}
allcv = pd.DataFrame()


# parameters
parameters = [{
  'classifier': [ensemble.RandomForestClassifier()],
  'classifier__n_estimators': [2, 4, 8, 16, 32, 64],
  'classifier__max_depth': [2, 4, 8, 16, 32, 64],
  'classifier__max_samples': [None, .25, .5, .75],
}]


# binary relevance - random forest


model = f.training_phase(problem_transform.BinaryRelevance(), parameters, x, y)
test_result = f.testing_phase(model, k, l)
test_result['model'] = 'binary_relevance'
test_results = test_result


models['br_rf'] = model
result = model.cv_results_
results['br_rf'] = result
result = pd.DataFrame(result)
f.drop_write(result, TODROP2, INFOFILE)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'binary_relevance'
best_results = pd.DataFrame(best_result)
allcv = allcv.append(best_results)


# classifier chains - random forest


model = f.training_phase(problem_transform.ClassifierChain(), parameters, x, y)
test_result = f.testing_phase(model, k, l)
test_result['model'] = 'classifier_chains'
test_results = test_results.append(test_result)


models['cc_rf'] = model
result = model.cv_results_
results['cc_rf'] = result
result = pd.DataFrame(result)
f.drop_write(result, TODROP2, INFOFILE)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'classifier_chains'
best_results = pd.DataFrame(best_result)
allcv = allcv.append(best_results)


# label powerset - random forest


model = f.training_phase(problem_transform.LabelPowerset(), parameters, x, y)
test_result = f.testing_phase(model, k, l)
test_result['model'] = 'label_powerset'
test_results = test_results.append(test_result)


models['lp_rf'] = model
result = model.cv_results_
results['lp_rf'] = result
result = pd.DataFrame(result)
f.drop_write(result, TODROP2, INFOFILE)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'label_powerset'
best_results = pd.DataFrame(best_result)
allcv = allcv.append(best_results)


# best plot final scores for all models
print(test_results)
f.drop_write(test_results, TODROP2, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'model': list(test_results['model']),
    'accuracy': list(test_results['accuracy']),
    'f1_micro': list(test_results['f1_micro']),
    'hamming_loss': list(test_results['hamming_loss']),
  }),
  filename='../random_forest_base/all_models_all_final_scores.png',
  x='model',
  title='best final scores on test set - all models - random forest base',
  ylabel='score'
)


# best plot cv test performance for all models
print(allcv)
f.drop_write(allcv, TODROP2, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'model': list(allcv['model']),
    'accuracy': list(allcv['mean_test_accuracy']),
    'f1_micro': list(allcv['mean_test_f1_micro']),
    'hamming_loss': list(allcv['mean_test_hamming_loss']),
  }),
  filename='../random_forest_base/all_models_all_cv_test_scores.png',
  x='model',
  title='best cross validation performance - all models - random forest base - test',
  ylabel='score'
)
# best plot cv train performance for all models
print(allcv)
f.drop_write(allcv, TODROP2, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'model': list(allcv['model']),
    'accuracy': list(allcv['mean_train_accuracy']),
    'f1_micro': list(allcv['mean_train_f1_micro']),
    'hamming_loss': list(allcv['mean_train_hamming_loss']),
  }),
  filename='../random_forest_base/all_models_all_cv_train_scores.png',
  x='model',
  title='best cross validation performance - all models - random forest base - train',
  ylabel='score'
)
