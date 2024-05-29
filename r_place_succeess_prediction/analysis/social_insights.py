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
from os.path import join as opj
import shap
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
if sys.platform == 'linux':
    sys.path.append('/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.success_level_prediction.success_level_predictions_utils import *
from r_place_success_analysis.success_level_prediction.dataframe_creator import DataFrameCreator

# configurations
analysis_type = 'two_models_comparison'#'single_model' #'two_models_comparison'
# if set to None - all relevant srs are taken into account. This is useful for debugging
maximum_srs = None

#elonmusk        0.996462
#kirby           0.993075
#greenlantern    0.985913

if __name__ == "__main__":
    if analysis_type == 'single_model':
        # configurations
        model_to_analyze = '3.21'
        path_to_config_file = opj("/sise/home/isabrah/reddit_canvas/results/success_analysis", model_to_analyze)
        config_dict = commentjson.load(open(opj(path_to_config_file, 'config_dict.json')))
        machine = 'yalla' if sys.platform == 'linux' else os.environ['COMPUTERNAME']
        target_data_path = opj(config_dict['target_data_dir'][machine], config_dict['target_data_f_name'])
        sr_objs_path = config_dict['sr_objs_path'][machine]

        validate_config(config_dict=config_dict, is_binary_class=False, check_model_existence=False)
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
        y_vec = dfc_obj.create_y_feature(normalize=False, log_scale=eval(config_dict['target_feature']['log_norm']))
        # removing communities that vanished and had zero pixels
        y_vec = {name: value for name, value in y_vec.items() if value != 0}
        y_df = pd.DataFrame.from_dict(y_vec, orient='index', columns=['label'])
        x_df = dfc_obj.create_x_matrix(verbose=False)
        modeling_df = pd.merge(x_df, y_df, left_index=True, right_index=True)
        print(f"Here is the distribution of the target feature: {Counter(modeling_df['label'])}")
        # randomizing the rows
        modeling_df_imputed = modeling_df.sample(frac=1, random_state=config_dict['random_seed'])
        # pulling out the explanatory features
        explain_features = [c for c in modeling_df.columns if c != 'label']


        # starting the actual modeling - we run a cv process (might also be be loo)
        true_values_and_predictions = list()
        rmse_r2_per_fold = dict()
        x_train = modeling_df[explain_features]
        y_train = modeling_df['label']

        # Data perp - filling missing values
        x_train_imputed = x_train.copy()
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(X=x_train)
        x_train_imputed[:] = imp_mean.transform(X=x_train)

        # Data perp - scaling (simple one)
        x_train_imputed_scaled = x_train_imputed.copy()
        scaler = StandardScaler()
        scaler.fit(X=x_train_imputed)
        x_train_imputed_scaled[:] = scaler.transform(X=x_train_imputed)

        clf_name = eval(config_dict['regression_model']['reg'])
        clf_params = config_dict['regression_model']['reg_params']
        cur_clf = clf_name(**clf_params)
        cur_clf.fit(X=x_train_imputed_scaled, y=y_train)
        current_train_prediction = cur_clf.predict(X=x_train_imputed_scaled)
        # some of the predictions appear to be negative, we will convert them to zero
        current_train_prediction = [0 if cp < 0 else cp for cp in current_train_prediction]

        # train eval measures
        cur_train_rmse = sqrt(mean_squared_error(y_train, current_train_prediction))
        cur_train_r2 = r2_score(y_train, current_train_prediction)

        # feature importance for the model
        importances = cur_clf.feature_importances_
        indices = np.flip(np.argsort(importances))
        features = x_train_imputed_scaled.columns
        feature_importance_sorted = {f: v for f, v in zip(features[indices], importances[indices])}
        print(f"Here is the list of the most important features, sorted: \n")
        print(feature_importance_sorted)

        # SHAP analysis for all observations (currently using the KMeans option - for faster process)
        clf_shap_explainer = shap.KernelExplainer(cur_clf.predict, shap.kmeans(x_train_imputed_scaled, 20))
        clf_shap_values = clf_shap_explainer.shap_values(x_train_imputed_scaled)
        plt.clf()
        shap.summary_plot(clf_shap_values, x_train_imputed_scaled, show=False)
        # editing the fig size
        _, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(h * 4 / 3, h)
        plt.savefig('summary_plot.png', bbox_inches='tight')
        # SHAP analysis for a single observation (elonmask here)
        clf_shap_explainer = shap.KernelExplainer(cur_clf.predict, x_train_imputed_scaled)
        clf_shap_values = clf_shap_explainer.shap_values(x_train_imputed_scaled[x_train_imputed_scaled.index=='elonmusk'])
        clf_shap_values_dict = {f: shap_v for f, shap_v in zip(x_train_imputed_scaled.columns, clf_shap_values[0])}
        shap.force_plot(clf_shap_explainer.expected_value, clf_shap_values[0, :],
                        x_train_imputed_scaled[x_train_imputed_scaled.index=='elonmusk'],
                        show=False, matplotlib=True)
        plt.savefig('force_plot.png')

    elif analysis_type == 'two_models_comparison':
        model_a = '3.01'
        model_b = '3.21'
        path_to_model_a_res = opj("/sise/home/isabrah/reddit_canvas/results/success_analysis", model_a)
        path_to_model_b_res = opj("/sise/home/isabrah/reddit_canvas/results/success_analysis", model_b)
        model_a_res = pd.read_csv(opj(path_to_model_a_res, 'raw_level_res.csv'), index_col=0)
        model_a_res['diff'] = model_a_res['true_value'] - model_a_res['pred']
        model_b_res = pd.read_csv(opj(path_to_model_b_res, 'raw_level_res.csv'), index_col=0)
        model_b_res['diff'] = model_b_res['true_value'] - model_b_res['pred']
        models_joined = model_a_res.join(model_b_res, how='inner', lsuffix='_model_a', rsuffix='_model_b')
        # the improvement % of model_b, compared to model_a (it might also be a negative num, if model_a is better)
        models_joined['model_b_improvement'] = (abs(models_joined['diff_model_a']) -
                                                abs(models_joined['diff_model_b'])) / abs(models_joined['diff_model_a'])
        models_joined.sort_values(by='model_b_improvement', ascending=False, inplace=True)
        print(f"Here is the list of the most interesting communities, sorted: \n{models_joined[0:10]['model_b_improvement']}")
        # now, we can use the notebooks and the SHAP package to create figures of those specific communities



