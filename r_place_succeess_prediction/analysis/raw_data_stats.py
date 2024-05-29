import re
import multiprocessing as mp
import sys
import pickle
if sys.platform == 'linux':
    sys.path.append('/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.success_level_prediction.success_level_predictions_utils import *
pd.set_option('display.max_columns', None)
# configurations
sr_objs_path = "/sise/Yalla_work/data/reddit_place/canvas_annotation_effort_data/concise_drawing_sr_objects"
path_to_target_df = "/sise/Yalla_work/data/reddit_place/canvas_annotation_effort_data/success_analysis_target_df_25_08_2022.p"
# if set to None - all relevant srs are taken into account. This is useful for debugging
debug_mode = False
maximum_srs = None


def extract_com_names(maximum_srs=None):
    com_level_gold_label_df = pickle.load(open(path_to_target_df, "rb"))
    # taking only srs which were manually labeled (~300 srs exist but did not really participate in r/place)
    com_level_gold_label_df = com_level_gold_label_df[com_level_gold_label_df['manually_labeled']].copy()
    # taking only the communities that existed BEFORE the r/place experiment
    sr_names = list(com_level_gold_label_df[com_level_gold_label_df['is_created_for_rplace'] == False].index)
    sr_names = sorted(sr_names)
    # filtering some SRs - if required
    if maximum_srs is not None:
        sr_names = sr_names[0:maximum_srs]
    return sr_names


def _pull_ststs_per_sr(sr_obj_explicit_path, sr_name, verbose=False):
    cur_sr_data_dict = dict()
    cur_sr_obj = pickle.load(open(sr_obj_explicit_path, "rb"))
    # pulling all relevant features BEFORE the experiment
    cur_sr_data_dict['num_submissions'] = cur_sr_obj.meta_features[0]['submission_amount']
    try:
        cur_sr_data_dict['num_comments'] = cur_sr_obj.meta_features[0]['comments_amount']
    except KeyError:
        cur_sr_data_dict['num_comments'] = 0
    cur_sr_data_dict['num_users'] = cur_sr_obj.meta_features[0]['users_amount']
    cur_sr_data_dict['days_pazam'] = cur_sr_obj.meta_features[0]['days_pazam']
    try:
        cur_sr_data_dict['num_upvotes'] = cur_sr_obj.meta_features[0]['comments_average_score']*cur_sr_obj.meta_features[0]['comments_amount'] + \
                                          cur_sr_obj.meta_features[0]['submission_average_score']*cur_sr_obj.meta_features[0]['submission_amount']
    except KeyError:
        cur_sr_data_dict['num_upvotes'] = cur_sr_obj.meta_features[0]['submission_average_score'] * cur_sr_obj.meta_features[0]['submission_amount']
    cur_sr_data_dict['num_tokens'] = sum(cur_sr_obj.submissions_tokens_dict.values()) + \
                                     sum(cur_sr_obj.comments_tokens_dict.values())
    # pulling all relevant features WHILE the experiment
    cur_sr_data_dict['exper_num_submissions'] = cur_sr_obj.exper_meta_features['submission_amount']
    try:
        cur_sr_data_dict['exper_num_comments'] = cur_sr_obj.exper_meta_features['comments_amount']
    except KeyError:
        cur_sr_data_dict['exper_num_comments'] = 0
    cur_sr_data_dict['exper_num_users'] = cur_sr_obj.exper_meta_features['users_amount']
    try:
        cur_sr_data_dict['exper_num_upvotes'] = cur_sr_obj.exper_meta_features['comments_average_score'] * \
                                          cur_sr_obj.exper_meta_features['comments_amount'] + \
                                          cur_sr_obj.exper_meta_features['submission_average_score'] * \
                                          cur_sr_obj.exper_meta_features['submission_amount']
    except KeyError:
        cur_sr_data_dict['exper_num_upvotes'] = cur_sr_obj.exper_meta_features['submission_average_score'] * \
                                          cur_sr_obj.exper_meta_features['submission_amount']
    cur_sr_data_dict['exper_num_tokens'] = sum(cur_sr_obj.exper_submissions_tokens_dict.values()) + \
                                           sum(cur_sr_obj.exper_comments_tokens_dict.values())
    return sr_name, cur_sr_data_dict


if __name__ == "__main__":
    sr_obj_f_names = sorted([f for f in os.listdir(sr_objs_path) if re.match(r'.*sr_obj_.*\.p', f)])
    com_names = extract_com_names(maximum_srs=maximum_srs)
    # taking only file names that exists in the list of sr_names of the object
    sr_obj_f_names = [sofn for sofn in sr_obj_f_names if sofn.split('sr_obj_')[-1].split('.p')[0] in com_names]
    path_of_each_sr_obj = [opj(sr_objs_path, sofn) for sofn in sr_obj_f_names]
    verbose_in_mp = True
    # the second element is the actual community name
    data_for_poll = [(poeso, poeso.split('/')[-1].split('.p')[0].split('sr_obj_')[-1], verbose_in_mp) for poeso in path_of_each_sr_obj]
    if debug_mode:
        pool = mp.Pool(processes=1)  # useful for debugging
    else:
        pool = mp.Pool(processes=int(mp.cpu_count() * 0.75 + 1))
    with pool as pool:
        results = pool.starmap(_pull_ststs_per_sr, data_for_poll)
    # now the results hold a tuple with the name of the SR in first place and dict of tokens in the second
    # we will convert is to a dataframe
    res_as_df = pd.DataFrame.from_dict({sr_name: feature for sr_name, feature in results}, orient='index')
    res_as_df.loc['total'] = res_as_df.sum()
    res_as_df.loc['mean'] = res_as_df.mean()
    res_as_df.loc['median'] = res_as_df.median()
    res_as_df.loc['std'] = res_as_df.std()
    print(res_as_df.loc[['total', 'mean', 'median', 'std']])
