import os
from glob import glob

import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from keras.layers import Conv2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten, MaxPool2D, BatchNormalization
from keras.models import Sequential
from sklearn.metrics import precision_recall_curve, auc, classification_report
from tqdm import tqdm

SAMPLE_SIZE = 5000

def ignore_rows(labels, image_name):
    # Using Anteroposterior 'View positions' only.
    return labels["View Position"][labels["Image Index"] == image_name].values[0] != "AP"

def create_imbalance(used_count, labels, image):
    # Creating imbalance by removing non-healthy images.
    return used_count % 3 == 0 and labels["Problem"][labels["Image Index"] == image].values[0] == 1
    # return False

def preProcess_Data_create_XY(images, labels, imba_set, imbaVP_set):
    x = [] # images as arrays
    y = [] # labels
    WIDTH = 128
    HEIGHT = 128
    count = 0
    used_count = 0
    total = images.__len__()

    loop = tqdm(total=len(images), position=0)
    for img in images:
        count += 1
        loop.set_description("Loading images".format(img))
        loop.update(1)
        base = os.path.basename(img)

        #  Creating imbalance by removing some of the examples for one class.
        if ((imba_set or imbaVP_set) and create_imbalance(used_count, labels, base)):
            continue
        # Attempt to simplify data-set. Using set View position.
        if (imbaVP_set and ignore_rows(labels, base)):
            continue

        used_count += 1

        # Read and resize image
        full_size_image = cv2.imread(img)
        # finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]
        problem = labels["Problem"][labels["Image Index"] == base].values[0]
        y.append(problem)
        x.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
    loop.close()
    print("Images used = ", used_count, " out of ", total)
    return x,y

def preProcess_prepare_XY(X, y):
    X = np.array(X)
    X = X / 255.0
    y = np.array(y)
    return X,y

def limit_sample_size(X_train, X_test, Y_train, Y_test):
    #Limit max samples, will not affect data in most cases.
    X_train = X_train[0:5000]
    Y_train = Y_train[0:5000]
    X_test = X_test[0:2000]
    Y_test = Y_test[0:2000]
    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)
    print("Training Data Shape:", len(X_train), X_train[0].shape)
    print("Testing Data Shape:", len(X_test), X_test[0].shape)
    return X_train, X_test, Y_train, Y_test

def imbalance_resolution(im, X_train, Y_train):
    X_for_method = np.array(X_train).reshape((len(X_train), 128 * 128 * 3))
    X_train, Y_train = im.fit_resample(X_for_method, Y_train)
    X_train = X_train.reshape((len(X_train), 128, 128, 3))
    return X_train, Y_train

def runCNN_And_Plot(x_train_data, x_test_data, y_train_data, y_test_data):
    # In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
    batch_size = 128
    input_shape = (128, 128, 3)
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                     activation ='relu', input_shape = input_shape,strides=8))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                     activation ='relu'))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same',
                     activation ='relu'))
    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same',
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = "sigmoid"))

    # Define the optimizer
    model.compile(optimizer="adam" , loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(x_train_data, y_train_data, epochs=25)


    y_pred_prob = model.predict(x_test_data)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred_prob]

    # Printing common score metrics FOR COMPARISON REASONS only.
    print(classification_report(y_test_data, y_pred))

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test_data, y_pred_prob, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plot_ROC(fpr, tpr, roc_auc)
    precision, recall, thresholds = precision_recall_curve(y_test_data, y_pred_prob)
    no_skill = len(y_test_data[y_test_data == 1]) / len(y_test_data)
    auc_score = auc(recall, precision)
    plot_Precision_Recall(precision=precision, recall=recall, no_skill=no_skill, auc=auc_score)

    plt.show()

def load_labels():
    labels = pd.read_csv('input/sample/sample_labels.csv')
    # keeping some of the unused columns
    labels = labels[['Image Index', 'Finding Labels', 'Patient Age', 'Patient Gender', 'View Position']]
    labels['Patient Age'] = labels['Patient Age'].apply(lambda x: x[:-1]).astype(int)
    labels['Problem'] = labels['Finding Labels'].apply(lambda x: 0 if 'No Finding' in x else 1)
    return labels

def load_images():
    multipleImages = glob('input/sample/sample/images/**')
    # Restricting sample size for testing reasons.
    multipleImages = multipleImages[:SAMPLE_SIZE]
    return multipleImages

def plot_images(multipleImages):
    i_ = 0
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for l in multipleImages[:25]:
        im = cv2.imread(l)
        im = cv2.resize(im, (128, 128))
        plt.subplot(5, 5, i_ + 1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));
        plt.axis('off')
        i_ += 1

def plot_label_distribution(X,y):
    df = pd.DataFrame()
    df["images"] = X
    df["labels"] = y
    lab = df['labels']
    plt.figure("Label distribution")
    sns.countplot(lab)

def plot_ROC(fpr, tpr, roc_auc):
    plt.figure("ROC")
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

def plot_Precision_Recall(precision, recall, no_skill, auc):
    plt.figure("Precision Recall Curve")
    plt.title('Precision Recall Curve')
    plt.plot(recall, precision, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.show()