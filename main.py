from utils.utils import create_directory
import os
import numpy as np
import sys
import sklearn
from utils.utils import load_dataset, load_dataset2, kfold_split, pre_train, load_dataset_test, reshape, save_experiment
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from itertools import product
import nni

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di registro di TensorFlow a solo errori

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  #Imposta il livello di registro di TensorFlow a solo errori

import warnings
warnings.filterwarnings("ignore")  #Ignora tutti i messaggi di avvertimento di Python

import keras
import logging
logging.getLogger('tensorflow').disabled = True  # Disabilita tutti i messaggi di log di TensorFlow

def fit_classifier(dataset, params):

    lr, mini_batch, tl = params
    X, y = dataset

    print(f'params: lr - {lr}, mini batch - {mini_batch}, tl - {tl}')
    

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []        

    for i, (train_index, val_index) in enumerate(kfold.split(X, y)):

      x_train, y_train, x_val, y_val = kfold_split(X, y, train_index, val_index)
      nb_classes = len(np.unique(np.concatenate((y_train, y_val), axis=0)))

      # transform the labels from integers to one hot vectors
      x_train, y_train, x_val, y_val, y_true = reshape(classifier_name, x_train, x_val, y_train, y_val)    

      input_shape = x_train.shape[1:]

      classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, lr = lr)

      if tl:
        pretrained_model = keras.models.load_model('../pretrained_models/resnet.hdf5')
        classifier.transfer_learning(pretrained_model)


      y_pred = classifier.fit(x_train, y_train, x_val, y_val, y_true, nb_epochs = 150, mini_batch_size=mini_batch)

      acc = accuracy_score(y_true, y_pred)
      print(f'fold: {i}, accuracy = {acc}')
      scores.append(acc)
    
    print(f'accuracy mean: {np.mean(scores)}, std: {np.std(scores)} \n\n')
    nni.report_intermediate_result(np.mean(scores))

    return np.mean(scores), np.std(scores)

def test_classifier():
    x_train, y_train = load_dataset2()
    x_test, y_test = load_dataset_test()

    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

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
    y_pred = classifier.fit(x_train, y_train, x_test, y_test, y_true, nb_epochs = 150)

    acc = accuracy_score(y_true, y_pred)
    print(f'test accuracy = {acc}')




def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False, lr = 0.01):
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose, lr=lr)
    if classifier_name == 'hivecote2':
        from classifiers import hivecote2
        return hivecote2.Classifier_HIVECOTE2(output_directory, input_shape, nb_classes, verbose)
    





classifier_name = sys.argv[1]
itr = sys.argv[2]

if itr == '_itr_0':
    itr = ''

output_directory = '../results/' + classifier_name + '/' + itr + '/'

test_dir_df_metrics = output_directory + 'df_metrics.csv'

print('Method: ', classifier_name, itr)
if os.path.exists(test_dir_df_metrics):
    print('Already done')
else:
    
    create_directory(output_directory)

    param_grid = {
    'learning_rate': 0.001,
    'mini_batch': 15,
    'transfer_learning': True 
    }

    optimized_params = nni.get_next_parameter()
    param_grid.update(optimized_params)

    # results = []

    # classifier_pretrained = create_classifier(classifier_name, (128,1), 4, output_directory, lr = 0.001)
    # pretrained_model = pre_train(output_directory,classifier_pretrained.model)

    dataset = load_dataset(split='TRAIN')

    #for idx, params in enumerate(param_combinations):
    #print(f'experiment: {idx}/{len(param_combinations)} \n')

    lr, mini_batch, tl = params = param_grid['learning_rate'], param_grid['mini_batch'], param_grid['transfer_learning']
    mean_acc, std_acc = fit_classifier(dataset, params)

    nni.report_final_result(mean_acc)

    # results.append([lr, mini_batch, tl, mean_acc, std_acc])
    
    #save_experiment(output_directory, results)

    
    print('DONE')
