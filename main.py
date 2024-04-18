from utils.utils import create_directory
import os
import numpy as np
import sys
import sklearn
from utils.utils import load_dataset2

def fit_classifier():

    x_train, y_train, x_test, y_test = load_dataset2()

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    
    # add a dimension to make it multivariate with one dimension 
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

    classifier.fit(x_train, y_train, x_test, y_test, y_true, nb_epochs = 150)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'hivecote2':
        from classifiers import hivecote2
        return hivecote2.Classifier_HIVECOTE2(output_directory, input_shape, nb_classes, verbose)
    





classifier_name = sys.argv[1]
itr = sys.argv[2]

if itr == '_itr_0':
    itr = ''

output_directory = 'results/' + classifier_name + '/' + itr + '/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', classifier_name, itr)
if os.path.exists(test_dir_df_metrics):
    print('Already done')
else:
    
    create_directory(output_directory)
    fit_classifier()
    print('DONE')
