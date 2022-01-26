# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 10.10.2021

# Example is taken from: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py
# the bert_as_a_service is really out of date - we will use newer representations of BERT and will not create new
# representations from BERT anymore
"""BERT finetuning runner."""
import sys
import os
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
import datetime
import commentjson
import sys
import r_place_drawing_classifier.pytorch_cnn.utils as pytorch_cnn_utils
import r_place_drawing_classifier.utils as r_place_drawing_classifier_utils
import r_place_drawing_classifier.pytorch_bert.utils as pytorch_bert_utils
from  r_place_drawing_classifier.pytorch_bert.bert_nn_model import BertNNModel
import pickle
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
import re
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from itertools import chain
import pandas as pd
import multiprocessing as mp


def _extract_sr_info(idx, sr_obj_file, data_path, net_feat_file):
    cur_sr = pickle.load(open(os.path.join(data_path, 'sr_objects', sr_obj_file), "rb"))
    # case the language of the current SR is French/Italian/Greek...
    if not (cur_sr.lang == 'en' or cur_sr.lang is None):
        print("SR {} was found with a foreign language "
              "(target={}), we skip it".format(cur_sr.name, cur_sr.trying_to_draw))
        return None
    res = cur_sr.meta_features_handler(smooth_zero_features=True,
                                       net_feat_file=net_feat_file,
                                       features_to_exclude=None)
    y_value = 1 if cur_sr.trying_to_draw == 1 else 0
    other_explanatory_features = \
        dict(cur_sr.explanatory_features) if eval(config_dict["meta_data_usage"]["use_meta"]) else dict()
    return {'sr_name': cur_sr.name, 'label': y_value, 'other_explanatory_features': other_explanatory_features}

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'config', 'modeling_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
data_path = config_dict['data_dir'][machine]
dict_input_as_args = [('-'+key, ) if type(value) is str and value in {'True', 'False'} else ('-'+key, str(value))
                      for key, value in config_dict['bert_config']['bert_server_params'].items()]
args_for_bert_server = list(chain(*dict_input_as_args))
#bert_server_args = get_args_parser().parse_args(args_for_bert_server)
########################################################################################################################
start_time = datetime.datetime.now()

pytorch_cnn_utils.set_random_seed(seed_value=config_dict["random_seed"])
# 'harambe' SR is problematic
if __name__ == "__main__":
    # update args of the configuration dictionary which can be known right as we start the run
    config_dict['machine'] = machine
    # finding all the SRs files
    sr_objects_path = os.path.join(data_path, 'sr_objects')
    sr_files = sorted([f for f in os.listdir(sr_objects_path) if re.match(r'sr_obj_.*\.p', f)])
    results_folder = os.path.join(config_dict['results_dir'][machine], 'model_'+config_dict['model_version'])
    # case the results folder doesn't exists, we'll create one
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    #bert_file_name = 'bert_embedding_model_' + config_dict['model_version'] + '.p'
    #bert_file_name = 'distilBert_representation_all_srs.p'
    bert_file_name = 'sentence_transformers_representation_all_srs.p'
    file_exists = os.path.isfile(os.path.join(results_folder, bert_file_name))
    srs_subm_embd = dict()
    y_vector = dict()
    other_explanatory_features = dict()
    # case the file representation exists, we'll just load it
    if file_exists:
        srs_subm_embd = pickle.load(open(os.path.join(results_folder, bert_file_name), "rb"))
    # network features usage (the ones Alex created)
    # case we want to add the network features to the explanatory features
    if eval(config_dict['meta_data_usage']['use_network']):
        net_feat_file = os.path.join(data_path, config_dict["meta_data_usage"]['network_file_path'][machine])
    else:
        net_feat_file = None
    dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
    # looping over all files in the folder, creating meta features + creating the embedding phase
    # we converted it to be a multiprocess instead of a simple loop
    processes_amount = 50
    input_for_pool = [(idx, f, data_path, net_feat_file) for idx, f in enumerate(sr_files)]
    pool = mp.Pool(processes=processes_amount)
    with pool as pool:
        results = pool.starmap(_extract_sr_info, input_for_pool)
    for r in results:
        if r is None:
            continue
        y_vector[r['sr_name']] = r['label']
        other_explanatory_features[r['sr_name']] = r['other_explanatory_features']
    """
    for file_idx, sr_obj_file in enumerate(sr_files):
        cur_sr = pickle.load(open(os.path.join(data_path, 'sr_objects', sr_obj_file), "rb"))
        # case the language of the current SR is French/Italian/Greek...
        if not (cur_sr.lang == 'en' or cur_sr.lang is None):
            print("SR {} was found with a foreign language "
                  "(target={}), we skip it".format(cur_sr.name, cur_sr.trying_to_draw))
            continue
        res = cur_sr.meta_features_handler(smooth_zero_features=True,
                                           net_feat_file=net_feat_file,
                                           features_to_exclude=None)
        # case the dict file with the embedding representation doesn't exist, we'll call the bert server
        if not file_exists:
            print("Problem along handling sr {}. Size mismatch".format(cur_sr.name))
        # if it does exist and the key we are looking for is in that dict
        elif cur_sr.name in srs_subm_embd:
            y_vector[cur_sr.name] = 1 if cur_sr.trying_to_draw == 1 else 0
            other_explanatory_features[cur_sr.name] = \
                dict(cur_sr.explanatory_features) if eval(config_dict["meta_data_usage"]["use_meta"]) else dict()
    """
    # creating a df for the training part (+evaluation)
    modeling_df = pytorch_bert_utils.build_modeling_df(explanatory_features=other_explanatory_features,
                                                       bert_embedding=srs_subm_embd,
                                                       normalize_explanatory=True, merge_method='inner',
                                                       fill_missing=True, value_for_missing=0)
    # end of loop
    eval_measures = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score,
                     'auc': roc_auc_score}

    eval_results = defaultdict(list)
    # running the classification model
    nn_x_input = torch.tensor([featrue_numeric_values.values for idx, featrue_numeric_values in sorted(modeling_df.iterrows())]).float()
    sr_names = [key for key, value in sorted(sorted(modeling_df.iterrows()))]
    print("Size of data before training: {}".format(nn_x_input.size()))
    nn_y_input = torch.tensor([value for key, value in sorted(y_vector.items()) if key in sr_names]).long()
    bert_modeling_obj = BertNNModel(input_dim=nn_x_input.shape[1], model=None, eval_measures=eval_measures, epochs=10, hid_size=100, dropout_perc=0.5,
                                    nonlin=F.relu, layers_amount=2, use_meta_features=False)
    cv_obj = StratifiedKFold(n_splits=config_dict['cv']['folds'], random_state=config_dict['random_seed'], shuffle=True)
    cv_obj.get_n_splits(nn_x_input, nn_y_input)
    all_test_data_pred = []
    for cv_idx, (train_index, test_index) in enumerate(cv_obj.split(nn_x_input, nn_y_input)):
        print("Fold {} starts".format(cv_idx))
        cur_train_data = nn_x_input[train_index]
        cur_test_data = nn_x_input[test_index]
        cur_y_train = nn_y_input[train_index]
        cur_y_test = nn_y_input[test_index]
        cur_test_sr_names = [sn for idx, sn in enumerate(sr_names) if idx in test_index]
        net = bert_modeling_obj.build_model(lr=0.1, suffle_train=True)
        net.fit(X=cur_train_data, y=cur_y_train)
        cur_test_predictions = net.predict_proba(X=cur_test_data)
        cur_results = bert_modeling_obj.calc_eval_measures(y_true=cur_y_test, y_pred=cur_test_predictions[:, 1],
                                                           nomalize_y=True)
        all_test_data_pred.extend([(int(y), pred, name) for name, y, pred in
                                   zip(cur_test_sr_names, cur_y_test, cur_test_predictions[:, 1])])
        duration = (datetime.datetime.now() - start_time).seconds
        print("Fold # {} has ended, updated results list is: {}. "
              "Current duration: {} seconds".format(cv_idx, cur_results, duration))

    print("Full modeling code has ended. Results are as follow: {}. \nThe process started at {}"
          " and finished at {}".format(bert_modeling_obj.eval_results, start_time, datetime.datetime.now()))
    # saving results to file if needed
    if eval(config_dict['saving_options']['measures']):
        results_file = os.path.join(config_dict['results_dir'][machine], config_dict['results_file'][machine])
        r_place_drawing_classifier_utils.save_results_to_csv(results_file=results_file, start_time=start_time,
                                                             objects_amount=len(sr_names), config_dict=config_dict,
                                                             results=bert_modeling_obj.eval_results)
    res_summary = all_test_data_pred

    # anyway, at the end of the code we will save results if it is required
    if eval(config_dict['saving_options']['raw_level_pred']):
        cur_folder_name = os.path.join(config_dict['results_dir'][machine], "model_" + config_dict['model_version'])
        if not os.path.exists(cur_folder_name):
            os.makedirs(cur_folder_name)
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
