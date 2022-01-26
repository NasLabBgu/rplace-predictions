# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 04.08.2019


import warnings
import sys
import re
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
import os
import datetime
import pickle
from r_place_drawing_classifier import utils as r_place_drawing_classifier_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import commentjson

from gensim.models.doc2vec import Doc2Vec
from collections import defaultdict
import numpy as np
import pandas as pd
from r_place_drawing_classifier.neural_net.pytorch_mlp import PytorchMLP
import torch
from torch.autograd import Variable
import random
from sklearn.preprocessing import StandardScaler, Imputer

warnings.simplefilter("ignore")
##################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'config', 'modeling_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
data_path = config_dict['data_dir'][machine]
########################################################################################################################


def calc_eval_measures(eval_measures, y_true, y_pred, nomalize_y=True):
    """
    calculation of the evaluation measures for a given prediciton vector and the y_true vector
    :param y_true: list of ints
        list containing the true values of y. Any value > 0 is considered as 1 (drawing),
        all others are 0 (not drawing)
    :param y_pred: list of floats
        list containing prediction values for each sr. It represnts the probability of the sr to be a drawing one
    :param nomalize_y: boolean. default: True
        whether or not to normalize the y_true and the predictions
    :return: dict
        dictionary with all the evalution measures calculated
    """
    eval_results = defaultdict(list)
    if nomalize_y:
        y_true = [1 if y > 0 else 0 for y in y_true]
        binary_y_pred = [1 if p > 0.5 else 0 for p in y_pred]
    else:
        binary_y_pred = [1 if p > 0.5 else -1 for p in y_pred]
    for name, func in eval_measures.items():
        eval_results[name].append(func(y_true, binary_y_pred))
    return eval_results


def set_random_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    random.seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    config_dict = r_place_drawing_classifier_utils.check_input_validity(config_dict=config_dict, machine=machine)
    # case a model was already built
    if os.path.exists(os.path.join(data_path, "x_as_df_doc2vec_DELETE_ME.p")):
        full_x_data_df = pickle.load(open(os.path.join(data_path, "x_as_df_doc2vec_DELETE_ME.p"), "rb"))
        y_vector = pickle.load(open(os.path.join(data_path, "y_vector_doc2vec_DELETE_ME.p"), "rb"))
    else:
        # first we need to load the sr objects
        sr_objects_path = os.path.join(data_path, 'sr_objects')
        sr_files = sorted([f for f in os.listdir(sr_objects_path) if re.match(r'sr_obj_.*\.p', f)])
        # setting random seed to all packages required
        set_random_seed(config_dict['random_seed'])
        # definition of the network features file (if required)
        if eval(config_dict['meta_data_usage']['use_network']):
            net_feat_file = os.path.join(data_path, config_dict["meta_data_usage"]['network_file_path'][machine])
        else:
            net_feat_file = None
        y_vector = dict()
        meta_dict = dict()
        network_dict = dict()
        doc2vec_dict = dict()
        com2vec_dict = dict()
        meta_df = pd.DataFrame()
        network_df = pd.DataFrame()
        doc2vec_df = pd.DataFrame()
        com2vec_df = pd.DataFrame()

        sr_names = set()
        missing_srs = defaultdict(list)
        # looping over each file and handling it
        for file_idx, sr_obj_file in enumerate(sr_files):
            cur_sr = pickle.load(open(os.path.join(data_path, 'sr_objects', sr_obj_file), "rb"))
            sr_names.add(cur_sr.name)
            if file_idx % 100 == 0 and file_idx > 0:
                duration = (datetime.datetime.now() - start_time).seconds
                print("Passed over {} SRs, time up to now: {} sec".format(file_idx, duration))
            # case the language of the current SR is French/Italian/Greek...
            # TODO: add ability to filter non English communities
            #if not (cur_sr.lang == 'en' or cur_sr.lang is None):
            y_vector[cur_sr.name] = 1 if cur_sr.trying_to_draw == 1 else -1
            res = cur_sr.meta_features_handler(smooth_zero_features=True,
                                               net_feat_file=net_feat_file,
                                               features_to_exclude=None)
            # case there was a problem with the function, we will remove the sr from the data
            if res != 0:
                missing_srs[cur_sr.name].append('meta_features')
            else:
                meta_dict[cur_sr.name] = {f_name: f_value for f_name, f_value in cur_sr.explanatory_features.items()
                                          if not f_name.startswith('network')}
                network_dict[cur_sr.name] = {f_name: f_value for f_name, f_value in cur_sr.explanatory_features.items()
                                             if f_name.startswith('network')}
        meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
        network_df = pd.DataFrame.from_dict(network_dict, orient='index')
        # end of loop
        # case we wish to use doc2vec representation
        if eval(config_dict['meta_data_usage']['use_doc2vec']):
            doc2vec_file = os.path.join(data_path, config_dict["meta_data_usage"]["doc2vec_file_path"][machine])
            doc2vec_model = Doc2Vec.load(doc2vec_file)
            # looping over all srs in the set we have, hopefully to find all of them in the model
            for cur_sr_name in sr_names:
                try:
                    doc2vec_dict[cur_sr_name] = {'doc2vec_'+str(value_idx): value for value_idx, value in enumerate(doc2vec_model.docvecs[cur_sr_name])}
                except KeyError:
                    missing_srs[cur_sr_name].append('doc2vec')
            doc2vec_df = pd.DataFrame.from_dict(doc2vec_dict, orient='index')
        if eval(config_dict['meta_data_usage']['use_com2vec']):
            com2vec_file = os.path.join(data_path, config_dict["meta_data_usage"]["com2vec_file_path"][machine])
            com2vec_model = pickle.load(open(com2vec_file, "rb"))
            # looping over all srs in the set we have, hopefully to find all of them in the model
            for cur_sr_name in sr_names:
                try:
                    cur_values_as_list = list(com2vec_model[cur_sr_name].values())
                    com2vec_dict[cur_sr_name] = {'com2vec_'+str(value_idx): value for value_idx, value in enumerate(cur_values_as_list)}
                except KeyError:
                    missing_srs[cur_sr_name].append('com2vec')
            com2vec_df = pd.DataFrame.from_dict(com2vec_dict, orient='index')

        # joins all data together in order to have one pandas df
        full_x_data_df = pd.concat([meta_df, network_df, doc2vec_df, com2vec_df], axis=1, join='inner')
        y_vector = pd.Series({key: value for key, value in y_vector.items() if key in full_x_data_df.index})
        # sorting the y vector according to the x one - since they must be aligned
        y_vector = y_vector[full_x_data_df.index]

        pickle.dump(full_x_data_df, open(os.path.join(data_path, "x_as_df_doc2vec_DELETE_ME.p"), "wb"))
        pickle.dump(y_vector, open(os.path.join(data_path, "y_vector_doc2vec_DELETE_ME.p"), "wb"))

    full_x_data_df = full_x_data_df.iloc[:, 68:]
    imp = Imputer(strategy='mean', copy=False)
    full_x_data_df = pd.DataFrame(imp.fit_transform(full_x_data_df), columns=full_x_data_df.columns,
                                  index=full_x_data_df.index)
    normalize_obj = StandardScaler()
    explanatory_features_df = pd.DataFrame(normalize_obj.fit_transform(full_x_data_df),
                                           columns=full_x_data_df.columns,
                                           index=full_x_data_df.index)

    duration = (datetime.datetime.now() - start_time).seconds
    print("Data preparation finished, shape of the x matrix is: {}. "
          "Took us up to now {} sec. Moving to modeling phases".format(full_x_data_df.shape, duration))
    eval_measures = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score,
                     'auc': roc_auc_score}
    kf = StratifiedKFold(n_splits=config_dict['cv']['folds'],
                         random_state=config_dict['random_seed'],
                         shuffle=True)
    # DELETE THIS SHIT LATER!!!! converts the vector to be [0,1] and not [-1,1]
    y_vector = y_vector.apply(lambda x: max(0, x))
    # building the models
    for fold_idx, (train_index, test_index) in enumerate(kf.split(full_x_data_df, y_vector)):
        print("starting fold {}".format(fold_idx))
        cur_fold_x_data_train = Variable(torch.tensor(full_x_data_df.iloc[train_index].values)).double()
        cur_fold_target_train = Variable(torch.tensor(y_vector.iloc[train_index].values)).type(torch.LongTensor)
        cur_fold_x_data_test = Variable(torch.tensor(full_x_data_df.iloc[test_index].values)).double()
        cur_fold_target_test = Variable(torch.tensor(y_vector.iloc[test_index].values)).type(torch.LongTensor)
        epochs_amount = config_dict['class_model']['nn_params']['epochs']
        # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)s
        model = PytorchMLP(full_x_data_df.shape[1], dropout=0.1, n_hid=128)
        model = model.double()
        optimizer = torch.optim.Adam(model.parameters())
        # create a loss function
        #criterion = torch.nn.SoftMarginLoss()
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs_amount):
            model.train()
            optimizer.zero_grad()  # zero the gradient buffer
            # since we are doing single instance batch optimization, we will loop over each
            # instance in the corpus

            for (cur_inst_x_data, cur_inst_target) in zip(cur_fold_x_data_train, cur_fold_target_train):
                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                outputs = model(cur_inst_x_data)
                # loss = criterion(outputs[:, 1], cur_inst_target)
                loss = criterion(outputs.unsqueeze(dim=0), cur_inst_target.unsqueeze(dim=0))
                if np.isnan(loss.tolist()):
                    print("wow")
                loss.backward()
                optimizer.step()
            """
            outputs = model(cur_fold_x_data_train)
            #loss = criterion(outputs[:, 1], cur_fold_target_train)
            loss = criterion(outputs, cur_fold_target_train)
            loss.backward()
            optimizer.step()
            """
            model.eval()
            with torch.no_grad():
                outputs = model(cur_fold_x_data_train)
                cur_epoch_prediction = outputs[:, 1].tolist()
                cur_epoch_eval_res_train = calc_eval_measures(eval_measures=eval_measures,
                                                              y_true=cur_fold_target_train.tolist(),
                                                              y_pred=cur_epoch_prediction, nomalize_y=True)

                print(f"finished epoch {epoch} (out of {epochs_amount}). Eval results: {cur_epoch_eval_res_train}")

        # train + test measures for this fold
        model.eval()
        with torch.no_grad():
            outputs = model(cur_fold_x_data_train)
            cur_fold_prediction = outputs[:, 1].tolist()
            cur_fold_eval_res_train = calc_eval_measures(eval_measures=eval_measures, y_true=cur_fold_target_train.tolist(),
                                                         y_pred=cur_fold_prediction, nomalize_y=True)
            # Test the Model using the test set
            outputs = model(cur_fold_x_data_test)
            cur_fold_prediction = outputs[:, 1].tolist()
            cur_fold_eval_res_test = calc_eval_measures(eval_measures=eval_measures, y_true=cur_fold_target_test.tolist(),
                                                        y_pred=cur_fold_prediction, nomalize_y=True)
            duration = (datetime.datetime.now() - start_time).seconds
            print("end of fold: {}. \nTrain eval results: {}\n"
                  "Test eval results: {}\n"
                  "Time up to now: {}".format(fold_idx, cur_fold_eval_res_train, cur_fold_eval_res_test, duration))

