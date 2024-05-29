import itertools
import commentjson
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, KFold
import sys
import pickle
from math import sqrt
if sys.platform == 'linux':
    sys.path.append('/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.success_level_prediction.success_level_predictions_utils import *
from r_place_success_analysis.success_level_prediction.dataframe_creator import DataFrameCreator

# configurations
path_to_config_file = "/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools/" \
                      "r_place_success_analysis/success_level_prediction"
config_dict = commentjson.load(open(opj(path_to_config_file, 'modeling_config.json')))
machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
target_data_path = opj(config_dict['target_data_dir'][machine], config_dict['target_data_f_name'])
sr_objs_path = config_dict['sr_objs_path'][machine]
# if set to None - all relevant srs are taken into account. This is useful for debugging
maximum_srs = None


if __name__ == "__main__":
    validate_config(config_dict=config_dict, is_binary_class=False)
    target_data_path_file = opj(target_data_path, config_dict['target_data_f_name'])
    t_feature = extract_target_feature(config_dict['target_feature'])
    dfc_obj = DataFrameCreator(target_feature=t_feature, path_to_srs_obj=sr_objs_path,
                               path_to_target_df=target_data_path,
                               while_or_before_exper=config_dict['while_or_before_exper'],
                               meta_features=eval(config_dict['features_usage']['use_meta']),
                               network_features=eval(config_dict['features_usage']['use_network']),
                               liwc=eval(config_dict['features_usage']['use_liwc']),
                               bow=eval(config_dict['features_usage']['use_bow']),
                               doc2vec=eval(config_dict['embeddings_usage']['use_doc2vec']),
                               com2vec=eval(config_dict['embeddings_usage']['use_com2vec']),
                               snap=eval(config_dict['embeddings_usage']['use_snap']),
                               graph2vec=eval(config_dict['embeddings_usage']['use_graph2vec']),
                               maximum_srs=maximum_srs,
                               debug_mode=eval(config_dict['debug_mode']),
                               bow_params=dict(config_dict['bow_params'])
                               )
    y_vec = dfc_obj.create_y_feature(use_percentile=eval(config_dict['target_feature']['use_percentile']),
                                     normalize=False, log_scale=eval(config_dict['target_feature']['log_norm']))
    # removing communities that vanished and had zero pixels
    y_vec = {name: value for name, value in y_vec.items() if value != 0}
    y_df = pd.DataFrame.from_dict(y_vec, orient='index', columns=['label'])
    x_df = dfc_obj.create_x_matrix(verbose=False)
    modeling_df = pd.merge(x_df, y_df, left_index=True, right_index=True)
    #print(f"Here is the distribution of the target feature: {Counter(modeling_df['label'])}")
    # randomizing the rows
    modeling_df = modeling_df.sample(frac=1, random_state=config_dict['random_seed'])
    # pulling out the explanatory features
    explain_features = [c for c in modeling_df.columns if c != 'label']

    # the next line is used to shrink the features to include ONLY the seniority one (to be used as a baseline model),
    # or the community_size one (to be used as a baseline model)
    #explain_features = ['days_pazam']#['days_pazam'] #['users_amount']

    # cross validation option (LOO or 10-fold-cv)
    if config_dict['cv']['folds'] == 'loo':
        cv_obj = LeaveOneOut()
    else:
        num_folds = config_dict['cv']['folds']
        cv_obj = KFold(n_splits=num_folds)

    # saving the modeling DF for future use
    if eval(config_dict['save_results']):
        machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
        model_ver = str(config_dict['model_version'])
        saving_full_path = opj(config_dict['results_dir'][machine], model_ver)
        if not os.path.exists(saving_full_path):
            os.makedirs(saving_full_path)
        pickle.dump(modeling_df, open(opj(saving_full_path, "modeling_df.p"), "wb"))
    # starting the actual modeling - we run a cv process (might also be be loo)
    true_values_and_predictions = list()
    rmse_r2_per_fold = dict()
    for loop_idx, (train_index, test_index) in enumerate(cv_obj.split(modeling_df[explain_features], modeling_df['label'])):
        # printing status (useful for the loo)
        if loop_idx > 0 and loop_idx % 100 == 0:
            print(f"passed over {loop_idx} cases so far")
        x_train, x_test = modeling_df[explain_features].iloc[train_index], modeling_df[explain_features].iloc[test_index]
        y_train, y_test = modeling_df['label'].iloc[train_index], modeling_df['label'].iloc[test_index]

        # Data perp - filling missing values
        x_train_imputed = x_train.copy()
        x_test_imputed = x_test.copy()
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(X=x_train)
        x_train_imputed[:] = imp_mean.transform(X=x_train)
        x_test_imputed[:] = imp_mean.transform(X=x_test)

        # Data perp - scaling (simple one)
        x_train_imputed_scaled = x_train_imputed.copy()
        x_test_imputed_scaled = x_test_imputed.copy()
        scaler = StandardScaler()
        scaler.fit(X=x_train_imputed)
        x_train_imputed_scaled[:] = scaler.transform(X=x_train_imputed)
        x_test_imputed_scaled[:] = scaler.transform(X=x_test_imputed)

        clf_name = eval(config_dict['regression_model']['reg'])
        clf_params = config_dict['regression_model']['reg_params']
        cur_clf = clf_name(**clf_params)
        cur_clf.fit(X=x_train_imputed_scaled, y=y_train)
        current_train_prediction = cur_clf.predict(X=x_train_imputed_scaled)
        current_test_prediction = cur_clf.predict(X=x_test_imputed_scaled)
        # some of the predictions appear to be negative, we will convert them to zero
        current_train_prediction = [0 if cp < 0 else cp for cp in current_train_prediction]
        current_test_prediction = [0 if cp < 0 else cp for cp in current_test_prediction]

        # the next line WAS used to get a baseline model (prediction based on prior distribution) - ZBBed
        #current_test_prediction = [np.mean(y_train) for i in current_test_prediction]
        # saving the results
        true_values_and_predictions.append((zip(list(x_test_imputed_scaled.index),
                                                list(y_test),
                                                list(current_test_prediction))))
        # we calculate RMSE, R^2 only if it is NOT a LOO setting
        if not config_dict['cv']['folds'] == 'loo':
            # train eval measures
            cur_train_rmse = sqrt(mean_squared_error(y_train, current_train_prediction))
            cur_train_r2 = r2_score(y_train, current_train_prediction)
            # test eval measures
            cur_test_rmse = sqrt(mean_squared_error(y_test, current_test_prediction))
            cur_test_r2 = r2_score(y_test, current_test_prediction)
            rmse_r2_per_fold[loop_idx] = {'n': len(y_test), 'train_rmse': cur_train_rmse, 'train_r2': cur_train_r2,
                                          'test_rmse': cur_test_rmse, 'test_r2': cur_test_r2}
    # unzipping the list of true+pred values
    true_values_and_predictions = [list(cur_zip) for cur_zip in true_values_and_predictions]
    true_values_and_predictions = list(itertools.chain(*true_values_and_predictions))
    true_values_and_predictions = {name: {'true_value': t_value, 'pred': pred}
                                   for name, t_value, pred in true_values_and_predictions}
    save_regression_res(config_dict=config_dict,
                        rmse_r2_per_fold=rmse_r2_per_fold,
                        true_values_and_predictions_raw_lvl=true_values_and_predictions,
                        verbose=True)
