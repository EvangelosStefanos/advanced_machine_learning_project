# Setting the random seeds for testing reproducibility.
from numpy.random import seed
from sklearn.model_selection import train_test_split

from CommonUtil import preProcess_Data_create_XY, runCNN_And_Plot, plot_images, load_images, load_labels, \
    preProcess_prepare_XY, plot_label_distribution, limit_sample_size, imbalance_resolution

seed(1)
import tensorflow
tensorflow.random.set_seed(2)

#############################################
def prepare_data(imba_set, imbaVP_set):
    images = load_images()
    plot_images(images)

    labels = load_labels()

    X, y = preProcess_Data_create_XY(images, labels, imba_set, imbaVP_set)
    plot_label_distribution(X, y)
    X, y = preProcess_prepare_XY(X, y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=666)
    limit_sample_size(X_train, X_test, Y_train, Y_test)
    return X_train, X_test, Y_train, Y_test


def run(imbalance_method, imba_set, imbaVP_set):
    X_train, X_test, Y_train, Y_test = prepare_data(imba_set, imbaVP_set)
    if (imbalance_method is not None):
        X_train, Y_train = imbalance_resolution(imbalance_method, X_train, Y_train)
    runCNN_And_Plot(X_train, X_test, Y_train, Y_test)