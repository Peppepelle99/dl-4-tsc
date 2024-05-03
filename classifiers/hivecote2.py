# hivecote2 model
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
matplotlib.use('agg')

from utils.utils import visualize_confusion_matrix

class Classifier_HIVECOTE2:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = HIVECOTEV2()
            print('model loaded')
        return
    
    def fit(self, x_train, y_train, x_val, y_val, y_true, mini_batch_size = 6, nb_epochs = 150):
        
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        start_time = time.time()

        print(f'start fit: ')
        hist = self.model.fit(x_train, y_train)
        duration = time.time() - start_time

        print(f'fit duration: {duration}')

        # y_pred = self.model.predict(x_train)
        # print(f'train accuracy: {accuracy_score(y_train, y_pred)}')

        y_val_pred = self.model.predict(x_val)
        #print(f'val accuracy: {accuracy_score(y_true, y_val_pred)}')

        visualize_confusion_matrix(self.output_directory, y_true, y_val_pred)
        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_val_pred)

        #df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return y_val_pred