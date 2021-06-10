import numpy as np
import pandas as pd
from scipy import io
from skmultilearn import problem_transform
import skmultilearn.model_selection
from sklearn import model_selection, svm, preprocessing,   ensemble, naive_bayes, neighbors, tree, metrics
import matplotlib.pyplot as plt


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


parameters_svc = [{
  'classifier': [svm.SVC()],
  'classifier__C': np.logspace(-4, 4, 9),
  'classifier__gamma' :  np.logspace(-4, 4, 9)
}]


def training_phase(model, params):
    model = model_selection.GridSearchCV(
      model,
      params,
      scoring={
          'accuracy': 'accuracy',
          'f1_macro': 'f1_macro',
          'f1_micro': 'f1_micro',
          'hamming_loss': metrics.make_scorer(metrics.hamming_loss)
      },
      refit='accuracy',
      return_train_score=True
    )


    model.fit(x, y)
    print (model.best_params_, model.best_score_)
    print(pd.DataFrame(model.cv_results_))
    return model


metric_names = ['accuracy', 'f1_micro', 'f1_macro', 'hamming_loss']


def testing_phase(model):
    l_star = model.predict(k)
    result = {}
    result['accuracy'] = [metrics.accuracy_score(l, l_star)]
    result['f1_micro'] = [metrics.f1_score(l, l_star, average='micro')]
    result['f1_macro'] = [metrics.f1_score(l, l_star, average='macro')]
    result['hamming_loss'] = [metrics.hamming_loss(l, l_star)]
    return result


def get_best_result(model):
    return pd.DataFrame(model.cv_results_).iloc[model.best_index_]


def get_metrics(result):
    res = {}
    for i in metric_names:
        res[i] = [result['mean_test_'+str(i)]]
    return res


models = {}
results = {}


# binary relevance - svms


model = training_phase(problem_transform.BinaryRelevance(), parameters_svc)


models['br_svc'] = model

result = model.cv_results_
results['br_svc'] = result

result = pd.DataFrame(result)

best_column = result.iloc[model.best_index_]
print(best_column)


data = {
    'classifier': ['svm'],
    'accuracy': best_column['mean_test_accuracy']
}
pd.DataFrame(data).plot.bar(x='classifier', rot=15)


data = {
    'classifier': ['svm'],
    'f1_macro': best_column['mean_test_f1_macro']
}
pd.DataFrame(data).plot.bar(x='classifier', rot=15)


d = pd.DataFrame()
d['param_classifier'] = results['br_6c']['param_classifier']
d['mean_test_accuracy_br'] = results['br_6c']['mean_test_accuracy']
d['mean_test_accuracy_cc'] = results['cc_6c']['mean_test_accuracy']
d['mean_test_accuracy_lp'] = results['lp_6c']['mean_test_accuracy']
d.plot.bar(x='param_classifier', rot=15)

