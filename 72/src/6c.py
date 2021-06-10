
# 6 classifiers

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


TODROP = ['split\d', 'time', 'std']

INFOFILE = '../multi_base/info.csv'
f.resetFile(INFOFILE)


x = data.x
y = data.y
k = data.k
l = data.l


models = {}
results = {}
allcv = pd.DataFrame()


# parameters
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



# binary relevance - multiple classifiers


model = f.training_phase(problem_transform.BinaryRelevance(), parameters_6_classifiers, x, y)
test_result = f.testing_phase(model, k, l)
test_result['model'] = 'binary_relevance'
test_results = test_result


models['br_6c'] = model
result = model.cv_results_
results['br_6c'] = result
result = pd.DataFrame(result)
result['model'] = 'binary_relevance'


# plot classifiers cv test scores
print(result)
f.drop_write(result, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'classifier': list(result['param_classifier']),
    'accuracy': list(result['mean_test_accuracy']),
    'f1_micro': list(result['mean_test_f1_micro']),
    'hamming_loss': list(result['mean_test_hamming_loss']),
  }),
  filename='../multi_base/binary_relevance_all_cv_test_scores.png',
  x='classifier',
  title='binary relevance - 6 base - all cv scores - test',
  ylabel='score'
)
# plot classifiers cv train scores
print(result)
f.drop_write(result, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'classifier': list(result['param_classifier']),
    'accuracy': list(result['mean_train_accuracy']),
    'f1_micro': list(result['mean_train_f1_micro']),
    'hamming_loss': list(result['mean_train_hamming_loss']),
  }),
  filename='../multi_base/binary_relevance_all_cv_train_scores.png',
  x='classifier',
  title='binary relevance - 6 base - all cv scores - train',
  ylabel='score'
)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'binary_relevance'
best_results = pd.DataFrame(best_result)
allcv = allcv.append(best_result)


# classifier chain - multiple classifiers


model = f.training_phase(problem_transform.ClassifierChain(), parameters_6_classifiers, x, y)
test_result = f.testing_phase(model, k, l)
test_result['model'] = 'classifier_chain'
test_results = test_results.append(test_result)

models['cc_6c'] = model
result = model.cv_results_
results ['cc_6c']= result
result = pd.DataFrame(result)
result['model'] = 'classifier_chains'



# plot classifiers cv test scores
print(result)
f.drop_write(result, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'classifier': list(result['param_classifier']),
    'accuracy': list(result['mean_test_accuracy']),
    'f1_micro': list(result['mean_test_f1_micro']),
    'hamming_loss': list(result['mean_test_hamming_loss']),
  }),
  filename='../multi_base/classifier_chains_all_cv_test_scores.png',
  x='classifier',
  title='classifier_chains - 6 base - all cv scores - test',
  ylabel='score'
)
# plot classifiers cv train scores
print(result)
f.drop_write(result, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'classifier': list(result['param_classifier']),
    'accuracy': list(result['mean_train_accuracy']),
    'f1_micro': list(result['mean_train_f1_micro']),
    'hamming_loss': list(result['mean_train_hamming_loss']),
  }),
  filename='../multi_base/classifier_chains_all_cv_train_scores.png',
  x='classifier',
  title='classifier_chains - 6 base - all cv scores - train',
  ylabel='score'
)


# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'classifier_chain'
best_results = best_results.append(best_result)
allcv = allcv.append(best_result)


# label powerset - multiple classifiers


model = f.training_phase(problem_transform.LabelPowerset(), parameters_6_classifiers, x, y)
test_result = f.testing_phase(model, k, l)
test_result['model'] = 'label_powerset'
test_results = test_results.append(test_result)

models['lp_6c'] = model
result = model.cv_results_
results['lp_6c'] = result
result = pd.DataFrame(result)
result['model'] = 'label_powerset'


# plot classifiers cv test scores
print(result)
f.drop_write(result, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'classifier': list(result['param_classifier']),
    'accuracy': list(result['mean_test_accuracy']),
    'f1_micro': list(result['mean_test_f1_micro']),
    'hamming_loss': list(result['mean_test_hamming_loss']),
  }),
  filename='../multi_base/label_powerset_all_cv_test_scores.png',
  x='classifier',
  title='label_powerset - 6 base - all cv scores - test',
  ylabel='score'
)
# plot classifiers cv train scores
print(result)
f.drop_write(result, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'classifier': list(result['param_classifier']),
    'accuracy': list(result['mean_train_accuracy']),
    'f1_micro': list(result['mean_train_f1_micro']),
    'hamming_loss': list(result['mean_train_hamming_loss']),
  }),
  filename='../multi_base/label_powerset_all_cv_train_scores.png',
  x='classifier',
  title='label_powerset - 6 base - all cv scores - train',
  ylabel='score'
)

# record best performance
best_result = f.get_best_result(model)
best_result['model'] = 'label_powerset'
best_results = best_results.append(best_result)
allcv = allcv.append(best_result)


print(best_results)


# plot best final scores for all models
print(test_results)
f.drop_write(test_results, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'model': list(test_results['model']),
    'accuracy': list(test_results['accuracy']),
    'f1_micro': list(test_results['f1_micro']),
    'hamming_loss': list(test_results['hamming_loss']),
  }),
  filename='../multi_base/all_models_all_final_scores.png',
  x='model',
  title='best final scores on test set - all models - multi base',
  ylabel='score'
)


# plot best cv performance for all models
print(allcv)
f.drop_write(allcv, TODROP, INFOFILE)
f.plot_and_save(
  data=pd.DataFrame({
    'model': list(allcv['model']),
    'accuracy': list(allcv['mean_test_accuracy']),
    'f1_micro': list(allcv['mean_test_f1_micro']),
    'hamming_loss': list(allcv['mean_test_hamming_loss']),
  }),
  filename='../multi_base/all_models_all_cv_scores.png',
  x='model',
  title='best cross validation performance - all models - multi base',
  ylabel='score'
)
