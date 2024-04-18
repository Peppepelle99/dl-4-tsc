# hivecote2 model
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
from utils.utils import save_test_duration

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.utils import save_logs
from utils.utils import calculate_metrics

class Classifier_HIVECOTE2:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = HIVECOTEV2()
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_from_path(self.output_directory
                                        .replace('hivecote2_augment', 'hivecote2')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.zip')
            else:
                self.model.save(self.output_directory + 'model_init')
        return
    
    def fit(self, x_train, y_train, x_val, y_val, y_true, mini_batch_size = 6, nb_epochs = 150):
        
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        start_time = time.time()

        hist = self.model.fit(x_train, y_train)
        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.zip')

        y_pred = self.predict(x_val)
        print(accuracy_score(y_pred, y_true))

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        #df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return 