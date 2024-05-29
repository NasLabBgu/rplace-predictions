from os.path import join as opj
import pickle
import pandas as pd
import sys
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.data_creation.data_creation_utils import *

# configurations
target_data_dir = "/data/work/data/reddit_place/canvas_annotation_effort_data"
target_data_f_name = "success_analysis_modeling_df_31_01_2022.p"
data_path = "/data/work/data/reddit_place/canvas_annotation_effort_data"
modeling_df_f_name = "modeling_df_without_gold_label_31_01_2022.p"


def _generate_experiment_features(sr_obj):
    res_dict = dict()
    sub_with_rel_regex = len([cur_s[2] for cur_s in sr_obj.submissions_as_list
                              if type(cur_s[2]) is str and ('r/place' in cur_s[2] or 'pixel' in cur_s[2])])
    com_with_rel_regex = len([cur_c[1] for cur_c in sr_obj.comments_as_list
                              if type(cur_c[1]) is str and ('r/place' in cur_c[1] or 'pixel' in cur_c[1])])
    try:
        res_dict['r_place_regex_in_submissions'] = sub_with_rel_regex
        res_dict['r_place_regex_in_submissions_norm'] = sub_with_rel_regex / len(sr_obj.submissions_as_list)
    except ZeroDivisionError:
        res_dict['r_place_regex_in_submissions'] = 0.0
        res_dict['r_place_regex_in_submissions_norm'] = 0.0
    try:
        res_dict['r_place_regex_in_comments'] = com_with_rel_regex
        res_dict['r_place_regex_in_comments_norm'] = com_with_rel_regex / len(sr_obj.comments_as_list)
    except ZeroDivisionError:
        res_dict['r_place_regex_in_comments'] = 0.0
        res_dict['r_place_regex_in_comments_norm'] = 0.0
    return res_dict


if __name__ == "__main__":
    # loading the target features df - it most be created before this code runs!!!
    com_level_gold_label_df = pickle.load(open(opj(target_data_dir, target_data_f_name), "rb"))
    # filtering communities which *were not* manually labeled
    com_level_gold_label_df = com_level_gold_label_df[com_level_gold_label_df['manually_labeled']].copy()
    # looping over each sr and pulling the explanatory features
    explanatory_features_dict = dict()
    structural_features_dict = dict()
    experiment_duration_features_dict = dict()
    for cur_sr_name in com_level_gold_label_df.index:
        try:
            cur_drawing_sr_obj = pickle.load(
                open(opj(data_path, 'drawing_sr_objects', 'drawing_sr_obj_' + cur_sr_name + '.p'), "rb"))
            experiment_duration_features_dict[cur_sr_name] = _generate_experiment_features(cur_drawing_sr_obj)
            if cur_drawing_sr_obj.explanatory_features == (None,):
                explanatory_features_dict[cur_sr_name] = None
            else:
                explanatory_features_dict[cur_sr_name] = cur_drawing_sr_obj.explanatory_features[0]
            if cur_drawing_sr_obj.structural_features is None:
                structural_features_dict[cur_sr_name] = None
            else:
                cur_structural_features_dict = {key: value for key, value in
                                                cur_drawing_sr_obj.structural_features.items()
                                                if type(value) is not dict and type(value) is not defaultdict}
                cur_structural_features_dict.update(
                    {key + '_avg': value['avg'] for key, value in cur_drawing_sr_obj.structural_features.items()
                     if type(value) is dict})
                cur_structural_features_dict.update(
                    {key + '_num': value['num'] for key, value in cur_drawing_sr_obj.structural_features.items()
                     if type(value) is defaultdict})
                structural_features_dict[cur_sr_name] = cur_structural_features_dict
        except FileNotFoundError:
            explanatory_features_dict[cur_sr_name] = None
    # looping over all srs in 'sr_name_to_id_dict' (srs from atlas) and pulling the explanatory feature
    explanatory_features_df = pd.DataFrame.from_dict({key: value for key, value in explanatory_features_dict.items()
                                                      if value is not None}, orient='index')
    structural_features_df = pd.DataFrame.from_dict({key: value for key, value in structural_features_dict.items()
                                                     if value is not None}, orient='index')
    experiment_duration_features_df = \
        pd.DataFrame.from_dict({key: value for key, value in experiment_duration_features_dict.items()}, orient='index')
    modeling_df = pd.merge(explanatory_features_df, structural_features_df, left_index=True, right_index=True)
    # THIS IS ONLY IN CASE WE WANT TO USE FEATURES THAT WERE PULLED DURING THE EXPERIMENT!!!
    modeling_df = pd.merge(modeling_df, experiment_duration_features_df, left_index=True, right_index=True)
    # saving the df to a file
    pickle.dump(modeling_df, open(opj(data_path, modeling_df_f_name), "wb"))
    print(f"\nA modeling df of size {modeling_df.shape} was created. Saved under {opj(data_path, modeling_df_f_name)}")
    print("\nNow you can move to build a classification model using the code in 'success_level_prediction'. Good Luck!")
