from builtins import print
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import os

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from scipy.interpolate import interp1d
np.float = float

from aeon.datasets import load_classification


def load_dataset(split):
  data = np.load('archives/Dataset_Liquid_Complete.npy')
  X = data[:, :-1]  
  y = data[:, -1]   


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

  if split == 'TRAIN':
    return  X_train, y_train
  elif split == 'TEST':
    return X_test, y_test
  else:
    return X_train, y_train, X_test, y_test


def reshape(classifier_name, x_train, x_val, y_train, y_val):

    if classifier_name != 'hivecote2':
          enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
          enc.fit(np.concatenate((y_train, y_val), axis=0).reshape(-1, 1))
          y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
          y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

          # save orignal y because later we will use binary
          y_true = np.argmax(y_val, axis=1)

          x_train = x_train.reshape((x_train.shape[0],x_train.shape[1], 1))
          x_val = x_val.reshape((x_val.shape[0],x_val.shape[1], 1))
    else:
        y_true = y_val
    
        x_train = x_train.reshape((x_train.shape[0],1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0],1, x_val.shape[1])) 
    
    return x_train, y_train, x_val, y_val, y_true


def load_dataset2():

    data = np.load('archives/Dataset_Liquid_Water.npy')
    X = data[:, :-1]  
    y = data[:, -1]

    return X,y

def load_dataset_test():
    data = np.load('archives/Dataset_Liquid_Copper.npy')
    X = data[:, :-1]  
    y = data[:, -1]

    return X,y

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory





def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])

    n = x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length


def transform_to_same_length(x, n_var, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts

    return ucr_x



def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    #for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

def save_experiment(output_directory, results):
    columns = ['lr', 'mini_batch', 'transfer_learning', 'mean_acc', 'std_acc']

    # Creazione del DataFrame
    df_experiment = pd.DataFrame(results, columns=columns)

    df_experiment.to_csv(output_directory + 'df_experiment.csv', index=False)

def visualize_filter(root_dir):
    import tensorflow.keras as keras
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    filters = model.layers[1].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter + 0.5, color='gray', label='filter')
    for c in classes:
        c_x_train = x_train[np.where(y_train == c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]

        idx_c = int(c) - 1

        plt.plot(c_x_train[idx], color=colors[idx_c], label='class' + str(idx_c) + '-raw')
        plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[idx_c], label='class' + str(idx_c) + '-conv')
        plt.legend()

    plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')

    return 1

def visualize_confusion_matrix(output_directory, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # Genera un grafico della matrice di confusione utilizzando seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(3), yticklabels=range(3))
    plt.xlabel('Etichetta Predetta')
    plt.ylabel('Etichetta Vera')
    plt.title('Matrice di Confusione')
    plt.show()
    plt.savefig(output_directory+'/cm.png')

def kfold_split(X, y, train_index, test_index, normalization=True ):
    x_train = X[train_index]
    y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]

    #Z-score Normalization
    if normalization:
        
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    return x_train, y_train, x_test, y_test

def pre_train(output_folder, model):
    x_train, y_train = load_classification(name="TwoPatterns", split="TRAIN")
    x_test, y_test = load_classification(name="TwoPatterns", split="TEST")

    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    x_train = x_train.reshape((x_train.shape[0],x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0],x_test.shape[2], 1))

    model.fit(x_train, y_train, batch_size=6, epochs=30,
                              verbose=True, validation_data=(x_test, y_test))
    
    model.save(output_folder + '/pretrained_model.hdf5')
    
    return model