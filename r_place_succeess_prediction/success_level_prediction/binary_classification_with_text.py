import itertools
import commentjson
import socket
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
import datetime
if sys.platform == 'linux':
    sys.path.append('/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.success_level_prediction.success_level_predictions_utils import *
from r_place_success_analysis.success_level_prediction.dataframe_creator import DataFrameCreator
from r_place_success_analysis.success_level_prediction.bert_modeler import BertModeler

# configurations
path_to_config_file = "/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools/" \
                      "r_place_success_analysis/success_level_prediction"
config_dict = commentjson.load(open(opj(path_to_config_file, 'modeling_config.json')))
# machines should be either 'yalla' or 'slurm' - prefer not to run on local PC
if 'yalla' in socket.gethostname() or sys.platform == 'linux':
    machine = 'yalla'
else:
    machine = os.environ['COMPUTERNAME']
# saving the machine to the config dict
config_dict['machine'] = machine
target_data_path = opj(config_dict['target_data_dir'][machine], config_dict['target_data_f_name'])
sr_objs_path = config_dict['sr_objs_path'][machine]
# if set to None - all relevant srs are taken into account. This is useful for debugging
maximum_srs = None


if __name__ == "__main__":
    validate_config(config_dict=config_dict, is_binary_class=True)
    start_time = datetime.datetime.now()
    # creating a folder to the model (in case it does not exist
    saving_folder_name = opj(config_dict['results_dir'][machine], "model_" + config_dict['model_version'])
    if not os.path.exists(saving_folder_name):
        os.makedirs(saving_folder_name)
    else:
        raise IOError(f"Note!! You provided a model name which already exist ({config_dict['model_version']}). "
                      f"Either delete the folder or provide a different model name")
    # saving the config file to the folder
    file_path = os.path.join(saving_folder_name, 'config_model_' + config_dict['model_version'] + '.json')
    with open(file_path, 'w') as fp:
        commentjson.dump(config_dict, fp, indent=2)
    # data loading
    target_data_path_file = opj(target_data_path, config_dict['target_data_f_name'])
    y_feature = extract_target_feature(config_dict['target_feature'])
    dfc_obj = DataFrameCreator(target_feature=y_feature, path_to_srs_obj=sr_objs_path,
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
    modeling_textual_data = dfc_obj.extract_text_for_bert_from_all_sr_objs()
    none_values_textual_data = [key for key, value in modeling_textual_data.items() if value is None]
    print(f"Textual data fro BERT modeling has been extracted. Out of {len(modeling_textual_data)} extracted "
          f"communities, {len(none_values_textual_data)} have no textual data at all.")
    y_vec = dfc_obj.create_y_feature()
    y_df = pd.DataFrame.from_dict(y_vec, orient='index', columns=['label'])
    x_df = dfc_obj.create_x_matrix()
    modeling_df = pd.merge(x_df, y_df, left_index=True, right_index=True)
    print(f"Here is the distribution of the target feature: {Counter(modeling_df['label'])}")
    # randomizing the rows
    modeling_df_imputed = modeling_df.sample(frac=1, random_state=config_dict['random_seed'])
    # pulling out the explanatory features
    explain_features = [c for c in modeling_df.columns if c != 'label']
    # adding a column called 'text' to be used by the BERT data loader. If no text exist we fill it with ''
    # TODO: make sure we only take English textual content
    modeling_df_imputed['text'] = [modeling_textual_data[cur_sr_name] if cur_sr_name in modeling_textual_data else ''
                                   for cur_sr_name in modeling_df_imputed.index]

    clf_params = config_dict['class_model']['clf_params']
    is_multimodal = True if len(explain_features) > 0 else False
    bert_modeler_obj = BertModeler(model_name=clf_params['model_name'],
                                   batch_size=clf_params['batch_size'],
                                   max_epochs=clf_params['max_epochs'],
                                   saving_path=saving_folder_name,
                                   multimodal=is_multimodal)
    cv_obj = StratifiedKFold(n_splits=config_dict['cv']['folds'], random_state=config_dict['random_seed'], shuffle=True)
    print(f"Starting the CV process. is_multimodal set to: {is_multimodal}")
    # loop over each CV iteration (usually 5)
    for cv_idx, (train_index, test_index) in enumerate(cv_obj.split(modeling_df_imputed.index, modeling_df_imputed['label'])):
        print(f"Fold {cv_idx} starts")
        cur_train_data = modeling_df_imputed.iloc[train_index]
        cur_test_data = modeling_df_imputed.iloc[test_index]
        cur_test_sr_names = list(cur_test_data.index)

        # now splitting the train to train and validation
        cur_train_data, cur_val_data = train_test_split(cur_train_data, test_size=0.2,
                                                        random_state=config_dict['random_seed'],
                                                        stratify=cur_train_data['label'])
        # updating the saving folder name
        bert_modeler_obj.saving_path = opj(saving_folder_name, 'cv_'+str(cv_idx))
        # make sure the saving path exists. If not - I create one
        if not os.path.exists(bert_modeler_obj.saving_path):
            os.makedirs(bert_modeler_obj.saving_path)
        bert_modeler_obj.train(train_dataset=cur_train_data, val_dataset=cur_val_data)
        # after training ends, we run the prediction over the test set, evaluate it and save it to disk
        cur_pred_output = bert_modeler_obj.predict_and_eval(test_dataset=cur_test_data)
        saving_measures = {key: value for key, value in cur_pred_output['measures'].items() if
                           'runtime' not in key and 'second' not in key}
        bert_modeler_obj.save_results_to_csv(results_file=opj(saving_folder_name, 'test_measures.csv'),
                                             cv_index=cv_idx, start_time=start_time, objects_amount=len(cur_test_data),
                                             results=saving_measures)
        print(f"Fold {cv_idx} ended. Results were saved under a csv in {saving_folder_name}")
