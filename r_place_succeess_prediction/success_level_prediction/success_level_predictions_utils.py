import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from math import sqrt
import sys
import os
from os.path import join as opj
import json


def validate_config(config_dict, is_binary_class, check_model_existence=True):
    machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
    model_ver = str(config_dict['model_version'])
    # checking if the user is running a binary classification file with configuration set to a binary problem
    if is_binary_class and not eval(config_dict['target_feature']['binary_class']):
        raise IOError("This code is intended to model binary classification only! Try other code rather than this one")
    # checking if the user is running a regression with configuration set to a regression problem
    if not is_binary_class and eval(config_dict['target_feature']['binary_class']):
        raise IOError(
            "This code is intended to model regression problem only! Try other code rather than this one")
    # checking if the results folder exists already. If so we raise an error
    model_full_path = opj(config_dict['results_dir'][machine], model_ver)
    if check_model_existence and os.path.exists(model_full_path):
        raise FileExistsError(f"The model provided in the config file exists. Either provide a different model version,"
                              " or delete the folder in the following path: {model_full_path}")


def extract_target_feature(config_dict_target_f):
    # simplest case - binary classification
    if eval(config_dict_target_f['binary_class']):
        return 'binary'
    # continuous target feature, normalized by demand/complexity
    elif eval(config_dict_target_f['factorize_target_feature']):
        if config_dict_target_f['factorize_by'] == 'demand_area':
            return 'pixels_demand_normalized'
        elif config_dict_target_f['factorize_by'] == 'complexity':
            return 'pixels_entropy_normalized'
        elif config_dict_target_f['factorize_by'] == 'community_size':
            return 'pixels_community_size_normalized'
        elif config_dict_target_f['factorize_by'] == 'diameter':
            return 'pixels_diameter_normalized'
    # continuous target feature, not normalized at all
    else:
        return 'pixels'


def save_binary_class_res(config_dict, precision_recall_per_fold, true_values_and_predictions_raw_lvl, verbose=True):
    precision_recall_per_fold_df = pd.DataFrame.from_dict(precision_recall_per_fold, orient='index')
    precision_recall_per_fold_df.loc['mean'] = precision_recall_per_fold_df.mean()
    precision_recall_per_fold_df.loc['std'] = precision_recall_per_fold_df.std()
    true_values_and_predictions_df = pd.DataFrame.from_dict(true_values_and_predictions_raw_lvl, orient='index')
    # we can now calculate the agg measures. It differs between loo and x-fold-cv
    agg_true_values = list(true_values_and_predictions_df['true_value'])
    agg_proba_prediction = list(true_values_and_predictions_df['pred'])
    agg_binary_prediction = [1 if pv > 0.5 else 0 for pv in agg_proba_prediction]
    agg_auc = roc_auc_score(agg_true_values, agg_proba_prediction)
    agg_prec, agg_recall, agg_fscore, _ = \
        precision_recall_fscore_support(agg_true_values, agg_binary_prediction, average='macro')
    agg_res_as_df = pd.DataFrame.from_dict({'prec': agg_prec, 'recall': agg_recall,
                                            'f-score': agg_fscore, 'auc': agg_auc}, orient='index')
    # creating the folder for results, if it doesn't exist
    if eval(config_dict['save_results']):
        machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
        model_ver = str(config_dict['model_version'])
        saving_full_path = opj(config_dict['results_dir'][machine], model_ver)
        os.makedirs(saving_full_path)
        # saving the config file + raw level results (as csv) + the results per fold as csv
        with open(opj(saving_full_path, 'config_dict.json'), 'w') as outfile:
            json.dump(config_dict, outfile)
        true_values_and_predictions_df.to_csv(opj(saving_full_path, 'raw_level_res.csv'), index=True, sep=',')
        precision_recall_per_fold_df.to_csv(opj(saving_full_path, 'res_per_fold.csv'), index=True, sep=',')
        print(f"All results and config file were saved (if required) under the following path: {saving_full_path}\n\n")
    if verbose:
        print(f"\nHere are the central matrices of the current run, per all instances together:\n {agg_res_as_df}")
        print("Here are the central matrices of the current run, per fold:")
        print(precision_recall_per_fold_df)


def save_regression_res(config_dict, rmse_r2_per_fold, true_values_and_predictions_raw_lvl, verbose=True):
    rmse_r2_per_fold_df = pd.DataFrame.from_dict(rmse_r2_per_fold, orient='index')
    rmse_r2_per_fold_df.loc['mean'] = rmse_r2_per_fold_df.mean()
    rmse_r2_per_fold_df.loc['std'] = rmse_r2_per_fold_df.std()
    true_values_and_predictions_df = pd.DataFrame.from_dict(true_values_and_predictions_raw_lvl, orient='index')
    # we can now calculate the agg measures. It differs between loo and x-fold-cv
    agg_true_values = list(true_values_and_predictions_df['true_value'])
    agg_predictions = list(true_values_and_predictions_df['pred'])
    agg_rmse = sqrt(mean_squared_error(agg_true_values, agg_predictions))
    agg_r2 = r2_score(agg_true_values, agg_predictions)
    agg_res_as_df = pd.DataFrame.from_dict({'rmse': agg_rmse, 'r2': agg_r2}, orient='index')
    # creating the folder for results, if it doesn't exist
    if eval(config_dict['save_results']):
        machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
        model_ver = str(config_dict['model_version'])
        saving_full_path = opj(config_dict['results_dir'][machine], model_ver)
        # in case the path does not exist
        if not os.path.exists(saving_full_path):
            os.makedirs(saving_full_path)
        # saving the config file + raw level results (as csv) + the results per fold as csv
        with open(opj(saving_full_path, 'config_dict.json'), 'w') as outfile:
            json.dump(config_dict, outfile)
        true_values_and_predictions_df.to_csv(opj(saving_full_path, 'raw_level_res.csv'), index=True, sep=',')
        rmse_r2_per_fold_df.to_csv(opj(saving_full_path, 'res_per_fold.csv'), index=True, sep=',')
        print(f"All results and config file were saved (if required) under the following path: {saving_full_path}\n\n")
    if verbose:
        print(f"\nHere are the central matrices of the current run, per all instances together:\n {agg_res_as_df}")
        print("Here are the central matrices of the current run, per fold:")
        print(rmse_r2_per_fold_df)


def find_classification_optimal_thresh(y_true, pred, f_score_avg='weighted'):
    precision, recall, thresholds = precision_recall_curve(y_true, pred)
    # looping over each threshold
    best_f1 = None
    best_thresh = None
    for thresh in thresholds:
        cur_binary_pred = [1 if p >= thresh else 0 for p in pred]
        cur_f1_score = f1_score(y_true=y_true, y_pred=cur_binary_pred, average=f_score_avg)
        if best_f1 is None or cur_f1_score > best_f1:
            best_f1 = cur_f1_score
            best_thresh = thresh
    return best_thresh, best_f1


