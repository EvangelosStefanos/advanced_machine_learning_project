import numpy as np
import pandas as pd
from scipy import io
from skmultilearn import problem_transform
import skmultilearn.model_selection
from sklearn import model_selection, svm, preprocessing,   ensemble, naive_bayes, neighbors, tree, metrics
import matplotlib.pyplot as plt
import matplotlib
import time, datetime

start = time.time()


matplotlib.rcParams['figure.figsize'] = [16, 9]

metric_names = ['accuracy', 'f1_micro', 'f1_macro', 'hamming_loss']


# load data

data = io.loadmat('../data/CHD_49.mat')
x = data['data']
y = data['targets']

# transform labels to {0,1}

y = preprocessing.Binarizer().fit_transform(y)

# split data

x, y, k, l = skmultilearn.model_selection.iterative_train_test_split(
  x, y, test_size = 0.25
)

# scale data

scaler = preprocessing.MinMaxScaler()
scaler = scaler.fit(x)
x = scaler.transform(x)
k = scaler.transform(k)


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
      return_train_score=True,
      cv=10
    )


    model.fit(x, y)
    #print (model.best_params_, model.best_score_)
    #print(pd.DataFrame(model.cv_results_))
    return model


def testing_phase(model):
    l_star = model.predict(k)
    result = {}
    result['accuracy'] = [metrics.accuracy_score(l, l_star)]
    result['f1_micro'] = [metrics.f1_score(l, l_star, average='micro')]
    result['f1_macro'] = [metrics.f1_score(l, l_star, average='macro')]
    result['hamming_loss'] = [metrics.hamming_loss(l, l_star)]
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


def hard_plot_and_save(y, xlabels, ylabel, labels, nsamples, ngroups):
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
  return


models = {}
results = {}


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


model = training_phase(problem_transform.BinaryRelevance(), parameters_6_classifiers)
test_result = testing_phase(model)
test_result['model'] = 'binary_relevance'
test_results = test_result


models['br_6c'] = model
result = model.cv_results_
results['br_6c'] = result
result = pd.DataFrame(result)

# plot classifiers vs accuracy
plot_and_save(
  data=result.filter(regex='param_classifier|mean_test_accuracy|mean_train_accuracy'),
  filename='../br_6c_accuracy.png',
  x='param_classifier'
)

# plot classifiers vs hamming loss
plot_and_save(
  data=result.filter(regex='param_classifier|mean_test_hamming_loss|mean_train_hamming_loss'),
  filename='../br_6c_hamming_loss.png',
  x='param_classifier'
)

# record best performance
best_result = get_best_result(model)
best_result['model'] = 'binary_relevance'
best_results = pd.DataFrame(best_result)


# classifier chain - multiple classifiers


model = training_phase(problem_transform.ClassifierChain(), parameters_6_classifiers)
test_result = testing_phase(model)
test_result['model'] = 'classifier_chain'
test_results = test_results.append(test_result)

models['cc_6c'] = model
result = model.cv_results_
results ['cc_6c']= result
result = pd.DataFrame(result)

# plot classifiers vs accuracy
plot_and_save(
  data=result.filter(regex='param_classifier|mean_test_accuracy|mean_train_accuracy'),
  filename='../cc_6c_accuracy.png',
  x='param_classifier'
)

# plot classifiers vs hamming loss
plot_and_save(
  data=result.filter(regex='param_classifier|mean_test_hamming_loss|mean_train_hamming_loss'),
  filename='../cc_6c_hamming_loss.png',
  x='param_classifier'
)

# record best performance
best_result = get_best_result(model)
best_result['model'] = 'classifier_chain'
best_results = best_results.append(best_result)


# label powerset - multiple classifiers


model = training_phase(problem_transform.LabelPowerset(), parameters_6_classifiers)
test_result = testing_phase(model)
test_result['model'] = 'label_powerset'
test_results = test_results.append(test_result)

models['lp_6c'] = model
result = model.cv_results_
results['lp_6c'] = result
result = pd.DataFrame(result)

# plot classifiers vs accuracy
plot_and_save(
  data=result.filter(regex='param_classifier|mean_test_accuracy|mean_train_accuracy'),
  filename='../lp_6c_accuracy.png',
  x='param_classifier'
)

# plot classifiers vs hamming loss
plot_and_save(
  data=result.filter(regex='param_classifier|mean_test_hamming_loss|mean_train_hamming_loss'),
  filename='../lp_6c_hamming_loss.png',
  x='param_classifier'
)

# record best performance
best_result = get_best_result(model)
best_result['model'] = 'label_powerset'
best_results = best_results.append(best_result)

print(best_results)

# plot test cv accuracy for each of the 3 models
plot_and_save(best_results, '../cross_model_cv_accuracy.png', x='model', y='mean_test_accuracy')

# plot test set accuracy for each of the 3 models
plot_and_save(test_results, '../cross_model_test_accuracy.png', x='model', y='accuracy')

# plot test set hamming loss for each of the 3 models
plot_and_save(test_results, '../cross_model_test_hamming_loss.png', x='model', y='hamming_loss')

print(test_results['accuracy'])
print(best_results['mean_test_accuracy'])
print(test_results['accuracy'].shape)


# plot test set accuracy and cv test accuracy for each of the 3 models
d = pd.DataFrame()
d['test_accuracy'] = list(test_results['accuracy'])
d['cv_accuracy'] = list(best_results['mean_test_accuracy'])
d['model'] = list(best_results['model'])
plt.xlabel('model')
plt.ylabel('accuracy')
plot_and_save(d, '../test_cv_accuracy_3.png', x='model')

# plot test set hamming loss and cv test hamming loss for each of the 3 models
d = pd.DataFrame()
d['test_hamming_loss'] = list(test_results['hamming_loss'])
d['cv_test_hamming_loss'] = list(best_results['mean_test_hamming_loss'])
d['model'] = list(best_results['model'])
plt.xlabel('model')
plt.ylabel('hamming_loss')
plot_and_save(d, '../test_cv_hamming_loss_3.png', x='model')






# parameters svms
parameters_svc = [{
  'classifier': [svm.SVC()],
  'classifier__C': np.logspace(-4, 4, 9),
  'classifier__gamma' :  np.logspace(-4, 4, 9)
}]


# binary relevance - svms


model = training_phase(problem_transform.BinaryRelevance(), parameters_svc)
test_result = testing_phase(model)
test_result['model'] = 'binary_relevance_svms'
test_results = test_result


models['br_svc'] = model
result = model.cv_results_
results['br_svc'] = result
result = pd.DataFrame(result)


# record best performance
best_result = get_best_result(model)
best_result['model'] = 'binary_relevance_svms'
best_results = best_result


# classifier chains - svms


model = training_phase(problem_transform.ClassifierChain(), parameters_svc)
test_result = testing_phase(model)
test_result['model'] = 'classifier_chain_svms'
test_results = test_results.append(test_result)


models['br_svc'] = model
result = model.cv_results_
results['br_svc'] = result
result = pd.DataFrame(result)


# record best performance
best_result = get_best_result(model)
best_result['model'] = 'classifier_chain_svms'
best_results = best_result.append(best_result)


# label powerset - svms


model = training_phase(problem_transform.LabelPowerset(), parameters_svc)
test_result = testing_phase(model)
test_result['model'] = 'label_powerset_svms'
test_results = test_results.append(test_result)


models['br_svc'] = model
result = model.cv_results_
results['br_svc'] = result
result = pd.DataFrame(result)

# record best performance
best_result = get_best_result(model)
best_result['model'] = 'label_powerset_svms'
best_results = best_result.append(best_result)

# plot test set accuracy and cv test accuracy for each of the 3 models
d = pd.DataFrame()
d['test_set_accuracy'] = list(test_results['accuracy'])
d['cv_test_accuracy'] = list(best_results['mean_test_accuracy'])
d['model'] = list(best_results['model'])
plt.xlabel('model')
plt.ylabel('accuracy')
plot_and_save(d, '../test_cv_accuracy_3_svms.png', x='model')


print('run time, ', str(datetime.timedelta(seconds=(time.time() - start))), ', seconds')
print('\a')
