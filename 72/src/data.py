
# data

from scipy import io
import skmultilearn.model_selection
from sklearn import model_selection, preprocessing


# load data

data = io.loadmat('../data/CHD_49.mat')
x = data['data']
y = data['targets']

# transform labels to {0,1}

y = preprocessing.Binarizer().fit_transform(y)

# split data

x, y, k, l = skmultilearn.model_selection.iterative_train_test_split(
  x, y, test_size = .25
)

# scale data

scaler = preprocessing.MinMaxScaler()
scaler = scaler.fit(x)
x = scaler.transform(x)
k = scaler.transform(k)
