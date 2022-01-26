# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021

# case is based on git from: https://github.com/Shawn1993/cnn-text-classification-pytorch

#! /usr/bin/env python
import os
import sys
import datetime
import torch
import torchtext.data as data
import commentjson
import sys
import r_place_drawing_classifier.pytorch_cnn.model as model
import r_place_drawing_classifier.pytorch_cnn.train as train
import r_place_drawing_classifier.utils as r_place_drawing_classifier_utils
import r_place_drawing_classifier.pytorch_cnn.utils as pytorch_cnn_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from collections import defaultdict
import pandas as pd

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'config', 'pytorch_config.json')))
machine = '' # name of the machine to be used. This should be sync with the config file
data_path = config_dict['data_dir'][machine]

config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'config', 'pytorch_config.json')))
config_dict = r_place_drawing_classifier_utils.check_input_validity(config_dict=config_dict, machine=machine)
########################################################################################################################
start_time = datetime.datetime.now()
pytorch_cnn_utils.set_random_seed(seed_value=config_dict["random_seed"])

if __name__ == "__main__":
    # update args of the configuration dictionary which can be known right as we start the run
    config_dict['machine'] = machine
    config_dict['kernel_sizes'] = [int(k) for k in
                                   config_dict['class_model']['cnn_max_pooling_parmas']['kernel_sizes'].split(',')]
    print("\nParameters:")
    for attr, value in sorted(config_dict.items()):
        print("\t{}={}".format(attr.upper(), value))

    # load data
    print("\nLoading reddit data...")
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    sr_name_field = data.Field(lower=True, use_vocab=True, sequential=False)
    meta_data_field = data.RawField(preprocessing=None, postprocessing=None)
    meta_data_field.is_target = False  # Workaround to solve a bug in pytorch
    all_data_prediction = dict()
    all_eval_results = defaultdict(list)
    for fold_number in range(config_dict['cv']['folds']):
        start_time = datetime.datetime.now()
        print("\nFold {} starts now".format(fold_number))
        train_data, dev_data, test_data =\
            pytorch_cnn_utils.place_data(text_field=text_field, label_field=label_field,
                                         sr_name_field=sr_name_field, meta_data_field=meta_data_field,
                                         config_dict=config_dict, fold_number=fold_number, device=-1, repeat=False)
        duration = (datetime.datetime.now() - start_time).seconds
        print("Loading data for fold {} has finished. Took us: {} seconds".format(fold_number, duration))

        # update args of the configuration dictionary which can be known only now (after loading data)
        config_dict['embed_num'] = len(text_field.vocab)
        config_dict['class_num'] = len(label_field.vocab) - 1
        config_dict['cuda'] = torch.cuda.is_available() and config_dict['gpu_usage']['device'] == "gpu"
        if eval(config_dict['meta_data_usage']['use_meta']):
            config_dict['meta_features_dim'] = len(train_data.dataset.examples[0].meta_data)
        else:
            config_dict['meta_features_dim'] = 0
        # model
        eval_measures_dict = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score,
                              'auc': roc_auc_score}
        embedding_config = config_dict['embedding']
        embed_file = os.path.join(data_path, embedding_config['file_path'][machine])
        cnn = model.CNN_Text(config_dict=config_dict, text_field=text_field, embedding_file=embed_file,
                             eval_measures=eval_measures_dict)
        if eval(config_dict['snapshot']) is not None:
            print('\nLoading model from {}...'.format(config_dict['snapshot']))
            cnn.load_state_dict(torch.load(config_dict['snapshot']))

        # trying to use the GPU - if it is avilable and the config support it
        if config_dict['cuda']:
            torch.cuda.set_device(0)
            cnn = cnn.cuda()

        # train or predict
        if eval(config_dict['mode']['predict']):
            label = train.predict(config_dict['mode']['predict'], cnn, text_field, label_field, config_dict['mode']['predict'])
            print('\n[Text]  {}\n[Label] {}\n'.format(config_dict['mode']['predict'], label))
        elif eval(config_dict['mode']['test']):
            try:
                train.eval(test_data, cnn, config_dict)
            except Exception as e:
                print("\nSorry. The test dataset doesn't exist.\n")
        else:
            try:
                cur_test_proba = train.train(train_data, dev_data, test_data, cnn, config_dict, fold=fold_number)
                cnn.calc_eval_measures(y_true=[value[1] for key, value in cur_test_proba.items()],
                                       y_pred=[value[0][1] for key, value in cur_test_proba.items()])
                # updating the cumulative results of the measures and the actual prediction of each SR
                for key, value in cnn.eval_results.items():
                    all_eval_results[key].append(value[0])
                all_data_prediction.update(cur_test_proba)
                print("Here are the current results: {}".format(cnn.eval_results))
                #here we should evaluate the results and save them
                #print("Here are the length of results: {}".format(len(cur_test_proba)))
            except KeyboardInterrupt:
                print('\n' + '-' * 89)
                print('Exiting from training early')
    # end of training (over all folds)
    # saving results section
    if eval(config_dict['saving_options']['measures']):
        results_file = os.path.join(config_dict['results_dir'][machine], config_dict['results_file'][machine])
        r_place_drawing_classifier_utils.save_results_to_csv(results_file=results_file, start_time=start_time,
                                                             SRs_amount=len(all_data_prediction),
                                                             config_dict=config_dict,
                                                             results=all_eval_results, saving_path=os.getcwd())
    # anyway, at the end of the code we will save results if it is required
    if eval(config_dict['saving_options']['raw_level_pred']):
        cur_folder_name = os.path.join(config_dict['results_dir'][machine], "model_" + config_dict['model_version'])
        if not os.path.exists(cur_folder_name):
            os.makedirs(cur_folder_name)
        res_summary = [(value[1], value[0][1], key) for key, value in all_data_prediction.items()]
        res_summary_df = pd.DataFrame(res_summary, columns=['true_y', 'prediction_to_draw', 'sr_name'])
        res_summary_df.to_csv(os.path.join(cur_folder_name,
                                           ''.join(['results_summary_model', config_dict['model_version'], '.csv'])),
                              index=False)
    if eval(config_dict['saving_options']['configuration']):
        cur_folder_name = os.path.join(config_dict['results_dir'][machine], "model_" + config_dict['model_version'])
        if not os.path.exists(cur_folder_name):
            os.makedirs(cur_folder_name)
        file_path = os.path.join(cur_folder_name, 'config_model_' + config_dict['model_version'] + '.json')
        with open(file_path, 'w') as fp:
            commentjson.dump(config_dict, fp, indent=2)
