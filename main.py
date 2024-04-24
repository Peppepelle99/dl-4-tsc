from utils.utils import create_directory
import os
import numpy as np
import sys
import sklearn
from utils.utils import load_dataset, load_dataset2, kfold_split, pre_train
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di registro di TensorFlow a solo errori

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  #Imposta il livello di registro di TensorFlow a solo errori

import warnings
warnings.filterwarnings("ignore")  #Ignora tutti i messaggi di avvertimento di Python

import keras
import logging
logging.getLogger('tensorflow').disabled = True  # Disabilita tutti i messaggi di log di TensorFlow

def fit_classifier():

    X,y = load_dataset2()

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    classifier_pretrained = create_classifier(classifier_name, (150,1), 2, output_directory)
    pretrained_model = pre_train(classifier_pretrained.model)

    for i, (train_index, test_index) in enumerate(kfold.split(X,y)):

      x_train, y_train, x_test, y_test = kfold_split(X, y, train_index, test_index)

      nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

      # transform the labels from integers to one hot vectors

      if classifier_name != 'hivecote2':
          enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
          enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
          y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
          y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

          # save orignal y because later we will use binary
          y_true = np.argmax(y_test, axis=1)

          x_train = x_train.reshape((x_train.shape[0],x_train.shape[1], 1))
          x_test = x_test.reshape((x_test.shape[0],x_test.shape[1], 1))
      else:
          y_true = y_test
    
          x_train = x_train.reshape((x_train.shape[0],1, x_train.shape[1]))
          x_test = x_test.reshape((x_test.shape[0],1, x_test.shape[1]))      

      input_shape = x_train.shape[1:]

      classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)

      #print(f'classificatore creato: {classifier.model.layers[1].get_weights()[0][0][0] == pretrained_model.layers[1].get_weights()[0][0][0]}')

      classifier.transfer_learning(pretrained_model, input_shape, nb_classes)

      #print(f'TL effettuato: {classifier.model.layers[1].get_weights()[0][0][0] == pretrained_model.layers[1].get_weights()[0][0][0]}')


      y_pred = classifier.fit(x_train, y_train, x_test, y_test, y_true, nb_epochs = 150)

      acc = accuracy_score(y_true, y_pred)
      print(f'fold: {i}, accuracy = {acc}')
      scores.append(acc)
    
    print(f'accuracy mean: {np.mean(scores)}, std: {np.std(scores)}')


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
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
