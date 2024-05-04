import numpy as np
from utils.utils import kfold_split, reshape, load_dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import nni
import keras



def fit_classifier(dataset, params, classifier_name, output_directory):

    X, y = dataset

    print(params)
    

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []        

    for i, (train_index, val_index) in enumerate(kfold.split(X, y)):

      x_train, y_train, x_val, y_val = kfold_split(X, y, train_index, val_index)
      nb_classes = len(np.unique(np.concatenate((y_train, y_val), axis=0)))

      # transform the labels from integers to one hot vectors
      x_train, y_train, x_val, y_val, y_true = reshape(classifier_name, x_train, x_val, y_train, y_val)    

      input_shape = x_train.shape[1:]


      if classifier_name == 'resnet':

        from classifiers import resnet
         
        classifier = resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose=False, lr=params['learning_rate'])

        if params['transfer_learning']:
            pretrained_model = keras.models.load_model('../pretrained_models/resnet.hdf5')
            classifier.transfer_learning(pretrained_model)


        y_pred = classifier.fit(x_train, y_train, x_val, y_val, y_true, nb_epochs = params['num_epochs'], mini_batch_size=params['mini_batch'])

      else:

        classifier = create_classifier(classifier_name, params)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_val)

      acc = accuracy_score(y_true, y_pred)
      print(f'fold: {i}, accuracy = {acc}')
      scores.append(acc)
      nni.report_intermediate_result(acc)
    
    print(f'accuracy mean: {np.mean(scores)}, std: {np.std(scores)} \n\n')

    return np.mean(scores), np.std(scores)

def test_classifier(params, classifier_name, output_directory):
    x_train, y_train, x_test, y_test = load_dataset()

    print(params)

    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    x_train, y_train, x_test, y_test, y_true = reshape(classifier_name, x_train, x_test, y_train, y_test) 


    input_shape = x_train.shape[1:]

    if classifier_name == 'resnet':

        from classifiers import resnet
         
        classifier = resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose=False, lr=params['learning_rate'])

        if params['transfer_learning']:
            pretrained_model = keras.models.load_model('../pretrained_models/resnet.hdf5')
            classifier.transfer_learning(pretrained_model)


        y_pred = classifier.fit(x_train, y_train, x_test, y_test, y_true, nb_epochs = params['num_epochs'], mini_batch_size=params['mini_batch'])

    else:

        classifier = create_classifier(classifier_name, params)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
    

    acc = accuracy_score(y_true, y_pred)
    print(f'test accuracy = {acc}')

def create_classifier(classifier_name, params):
    if classifier_name == 'hivecote2':
        from aeon.classification.hybrid import HIVECOTEV2
        return HIVECOTEV2()
    if classifier_name == 'multiHydra':
        from aeon.classification.convolution_based import MultiRocketHydraClassifier

        return MultiRocketHydraClassifier(n_kernels=params['n_kernels'], n_groups=params['n_groups'], random_state=42)
    