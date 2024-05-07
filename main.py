from utils.utils import create_directory
import sys
from utils.utils import load_dataset, select_params, pre_train
from utils.train_test import fit_classifier, test_classifier, fit_ensamble
import nni

# remove info-warning

import tensorflow as tf
import warnings
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.get_logger().setLevel('ERROR')  
warnings.filterwarnings("ignore")  
logging.getLogger('tensorflow').disabled = True 


mode = 'TRAIN'
ensamble = False

if ensamble:
    classifiers= sys.argv[1:]

    print('Method: ', classifiers, mode)
else:
    classifier_name = sys.argv[1]
    print('Method: ', classifier_name, mode)
    #itr = sys.argv[2]



# if itr == '_itr_0':
#     itr = ''

output_directory = '../results/'
# test_dir_df_metrics = output_directory + 'df_metrics.csv'



    
#create_directory(output_directory)

if ensamble:
    param_grid = select_params(classifiers[0])

    params = []
    for c in classifiers:
        params.append(select_params(c))
else:
    param_grid = select_params(classifier_name)

optimized_params = nni.get_next_parameter()
param_grid.update(optimized_params)
# params = param_grid['learning_rate'], param_grid['mini_batch'], param_grid['transfer_learning'], param_grid['num_epochs']

# classifier_pretrained = create_classifier(classifier_name, (128,1), 4, output_directory, lr = 0.001)
# pretrained_model = pre_train(output_directory,classifier_pretrained.model)

if mode == 'TRAIN':

    dataset = load_dataset(split='TRAIN')
    if ensamble:
        mean_acc, std_acc = fit_ensamble(dataset, params, classifiers)
    else:
        mean_acc, std_acc = fit_classifier(dataset, param_grid, classifier_name, output_directory)

    nni.report_final_result(mean_acc)

elif mode == 'TEST':
    test_classifier(param_grid, classifier_name, output_directory)



print('DONE')
