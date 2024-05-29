import itertools
import commentjson
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import sys
if sys.platform == 'linux':
    sys.path.append('/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.success_level_prediction.success_level_predictions_utils import *
from r_place_success_analysis.success_level_prediction.dataframe_creator import DataFrameCreator

pd.set_option('display.max_columns', None)
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
    validate_config(config_dict=config_dict, is_binary_class=True)
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
    y_vec = dfc_obj.create_y_feature()
    y_df = pd.DataFrame.from_dict(y_vec, orient='index', columns=['label'])
    x_df = dfc_obj.create_x_matrix(verbose=False)
    modeling_df = pd.merge(x_df, y_df, left_index=True, right_index=True)
    print(f"Here is the distribution of the target feature: {Counter(modeling_df['label'])}")
    # randomizing the rows
    modeling_df_imputed = modeling_df.sample(frac=1, random_state=config_dict['random_seed'])
    # pulling out the explanatory features
    explain_features = [c for c in modeling_df.columns if c != 'label']
    # the next line is used to shrink the features to include ONLY the seniority one (to be used as a baseline model)
    #explain_features = ['days_pazam']

    # cross validation option (LOO or 10-fold-cv)
    if config_dict['cv']['folds'] == 'loo':
        cv_obj = LeaveOneOut()
    else:
        num_folds = config_dict['cv']['folds']
        cv_obj = StratifiedKFold(n_splits=num_folds)

    # starting the actual modeling - we run a cv process (might also be be loo)
    true_values_and_predictions = list()
    eval_measures_per_fold = dict()
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

        clf_name = eval(config_dict['class_model']['clf'])
        clf_params = config_dict['class_model']['clf_params']
        cur_clf = clf_name(**clf_params)
        cur_clf.fit(X=x_train_imputed_scaled, y=y_train)
        # train prediction. We also find the optimal threshold as the default 0.5 is not optimal
        current_train_proba_prediction = cur_clf.predict_proba(X=x_train_imputed_scaled)
        best_thresh, best_train_f_score = find_classification_optimal_thresh(y_true=y_train,
                                                                       pred=current_train_proba_prediction[:, 1])
        current_train_binary_prediction = [1 if p >= best_thresh else 0 for p in current_train_proba_prediction[:, 1]]
        # test prediction
        current_test_proba_prediction = cur_clf.predict_proba(X=x_test_imputed_scaled)
        best_thresh, best_test_f_score = find_classification_optimal_thresh(y_true=y_test,
                                                                            pred=current_test_proba_prediction[:, 1])
        current_test_binary_prediction = [1 if p >= best_thresh else 0 for p in current_test_proba_prediction[:, 1]]

        # these two lines are used to get a baseline model (prediction based on prior distribution)
        #current_test_proba_prediction[:, 0] = 1 - np.mean(y_train)
        #current_test_proba_prediction[:, 1] = np.mean(y_train)
        # saving the results
        true_values_and_predictions.append((zip(list(x_test_imputed_scaled.index),
                                                list(y_test),
                                                list(current_test_proba_prediction[:, 1]))))
        # we calculate precision, recall etc only if it is NOT a LOO setting (for both train and test
        if not config_dict['cv']['folds'] == 'loo':
            # train eval
            cur_train_prec, cur_train_recall, cur_train_f_score, _ = \
                precision_recall_fscore_support(y_train, current_train_binary_prediction, average='weighted')
            cur_train_auc_score = roc_auc_score(y_train, current_train_proba_prediction[:, 1])
            # test eval
            cur_test_prec, cur_test_recall, cur_test_f_score, _ = \
                precision_recall_fscore_support(y_test, current_test_binary_prediction, average='weighted')
            cur_test_auc_score = roc_auc_score(y_test, current_test_proba_prediction[:, 1])
            eval_measures_per_fold[loop_idx] = {'n': len(y_test), 'train_prec': cur_train_prec,
                                                'train_recall': cur_train_recall, 'train_f_score': cur_train_f_score,
                                                'train_auc': cur_train_auc_score, 'test_prec': cur_test_prec,
                                                'test_recall': cur_test_recall, 'test_f_score': cur_test_f_score,
                                                'test_auc': cur_test_auc_score}
    # unzipping the list of true+pred values
    true_values_and_predictions = [list(cur_zip) for cur_zip in true_values_and_predictions]
    true_values_and_predictions = list(itertools.chain(*true_values_and_predictions))
    true_values_and_predictions = {name: {'true_value': t_value, 'pred': pred}
                                   for name, t_value, pred in true_values_and_predictions}
    save_binary_class_res(config_dict=config_dict,
                          precision_recall_per_fold=eval_measures_per_fold,
                          true_values_and_predictions_raw_lvl=true_values_and_predictions,
                          verbose=True)
