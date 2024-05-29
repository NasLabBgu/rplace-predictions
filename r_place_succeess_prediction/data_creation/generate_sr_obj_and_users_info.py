import os
import pickle
import pandas as pd
import sys
import multiprocessing as mp
from os.path import join as opj
import datetime
if sys.platform == 'linux':
    sys.path.append('/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.data_creation.reddit_community import RedditCommunity
from r_place_success_analysis.data_creation.data_creation_utils import create_sr_name_to_id_dict, create_users_data_dict
from sr_classifier.reddit_data_preprocessing import RedditDataPrep

###### configurations
data_path = "/sise/Yalla_work/data/reddit_place/canvas_annotation_effort_data"
sr_objects_path = "/sise/Yalla_work/data/reddit_place/sr_objects"
generate_users_info_data = False
generate_sr_obj = True
overwrite_sr_obj = False
# the next variables are for the BERT data creation
bert_submissions_sampling = True
bert_sampling_logic = 'score'
bert_sampling_percentage = 0.8
bert_max_subm = 10000

# Don't set more than 20
cpu_n = 20

#########
user_subreddits_dict = pickle.load(
    open(os.path.join('/sise/Yalla_work/data/reddit_place', 'user_subreddits_dict.pkl'), "rb"))
graph_dict = pickle.load(open(os.path.join(os.path.dirname(data_path), 'graph_dict.pickle'), "rb"))
atlas_mapping = pd.read_csv(os.path.join(data_path, 'atlas_mapping.csv'))
sr_name_to_id_dict = create_sr_name_to_id_dict(atlas_mapping=atlas_mapping)
atlas_data_as_list = pickle.load(open(opj(data_path, 'atlas_data_as_list.p'), "rb"))
sr_obj_saving_path = opj(data_path, "drawing_sr_objects")
concise_sr_obj_saving_path = opj(data_path, "concise_drawing_sr_objects")

start_time = datetime.datetime.now()


def load_and_concise_community_obj(index, sr_name, verbose=True):
    func_start_time = datetime.datetime.now()
    existing_pickle_f_name = opj(sr_obj_saving_path, "drawing_sr_obj_"+sr_name+".p")
    saving_f_name = opj(concise_sr_obj_saving_path, "concise_drawing_sr_obj_" + sr_name + ".p")
    if not overwrite_sr_obj and os.path.exists(saving_f_name):
        print(f"Object for {sr_name} exists and you asked not to overwrite files. Delete it or change configurations")
        return sr_name, 0, 0
    # loading the object
    cur_community_obj = pickle.load(open(existing_pickle_f_name, "rb"))
    # removing heavy items from it
    cur_community_obj.comments_as_list = None
    cur_community_obj.comments_as_tokens = None
    cur_community_obj.submissions_as_list = None
    cur_community_obj.submissions_as_tokens = None
    cur_community_obj.text_data_for_bert = None
    cur_community_obj.exper_comments_as_list = None
    cur_community_obj.exper_comments_as_tokens = None
    cur_community_obj.exper_submissions_as_list = None
    cur_community_obj.exper_submissions_as_tokens = None
    cur_community_obj.exper_text_data_for_bert = None
    # saving the concise object
    pickle.dump(cur_community_obj, open(saving_f_name, "wb"))
    if verbose:
        print(f"A concise Object for {sr_name} has been created and saved under: {saving_f_name}", flush=True)
    return 0


def calc_and_save_community_obj(index, sr_name, verbose=True):
    func_start_time = datetime.datetime.now()
    saving_f_name = opj(sr_obj_saving_path, "drawing_sr_obj_"+sr_name+".p")
    # need to make sure the file does not exist - if we don't want to override existing ones
    if not overwrite_sr_obj and os.path.exists(saving_f_name):
        print(f"Object for {sr_name} exists and you asked not to overwrite files. Delete it or change configurations")
        return sr_name, 0, 0
    local_tiles_placement = pd.read_csv(os.path.join(data_path, 'tiles_with_explicit_user_names.csv'))
    # pulling out information about the artwork of the current community
    cur_sr_artwork_ids = sr_name_to_id_dict[sr_name]
    end_state = sum([atlas_mapping[atlas_mapping['atlas_idx'] == artwork_id[0]]['survived'].iloc[0]
                     for artwork_id in cur_sr_artwork_ids])
    end_state = 'survived' if end_state >= 1 else 'extinct'
    # pulling out only the paths that are relevant to the list of ids we found
    cur_sr_artwork_paths = {cur_atlas_dict['id']: cur_atlas_dict['path'] for cur_atlas_dict in atlas_data_as_list
                            if cur_atlas_dict['id'] in [cur_pair[0] for cur_pair in cur_sr_artwork_ids]}
    cur_sr_artwork_ids_and_paths = [{'id': cur_pair[0], 'percentage': cur_pair[1],
                                     'path': cur_sr_artwork_paths[cur_pair[0]]} for cur_pair in cur_sr_artwork_ids]

    # pulling out the information pre experiment (all of it should exist in saved files already)
    sr_obj_found = 1
    struct_features_pre_exper_found = 1
    try:
        struct_features_pre_exper = graph_dict[sr_name]
    except KeyError:
        struct_features_pre_exper = None
        struct_features_pre_exper_found = 0
    try:
        historical_sr_obj = pickle.load(open(os.path.join(sr_objects_path, 'sr_obj_' + sr_name + '_.p'), "rb"))
        meta_features_pre_exper = dict(historical_sr_obj.explanatory_features)
        cur_sr_creation_utc = historical_sr_obj.creation_utc
        cur_sr_num_users = historical_sr_obj.num_users
    except FileNotFoundError:
        historical_sr_obj = None
        sr_obj_found = 0
        meta_features_pre_exper = None
        cur_sr_creation_utc = None
        cur_sr_num_users = None

    # generating the community object (and saving it to disk)
    cur_community_obj = RedditCommunity(name=sr_name, creation_utc=cur_sr_creation_utc,
                                        survived=True if end_state == 'survived' else False,
                                        num_users=cur_sr_num_users,
                                        meta_features_pre_exper=meta_features_pre_exper,
                                        struct_features_pre_exper=struct_features_pre_exper,
                                        pixels_path=cur_sr_artwork_ids_and_paths)
    # adding the new object all the textual information from the historical SR object (if exists)
    if historical_sr_obj is not None:
        cur_community_obj.extract_pre_exper_textual_data(pre_exper_sr_obj=historical_sr_obj)
    # adding all information that is related to the users
    cur_community_obj.extract_seed_users(tiles_placement=local_tiles_placement,
                                         user_subreddits_dict=user_subreddits_dict, consider_only_writting_users=False)
    cur_community_obj.extract_potential_users(tiles_placement=local_tiles_placement,
                                              user_subreddits_dict=user_subreddits_dict)

    # extracting the textual data that were created **during** the experiment
    if verbose:
        print(f"\nThread idx {index} reports: starting the textual data preprocessing for subreddit {sr_name}.", flush=True)
    submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
    comments_dp_obj = RedditDataPrep(is_submission_data=False, remove_stop_words=False, most_have_regex=None)
    cur_community_obj.extract_while_exper_textual_data(submission_dp_obj=submission_dp_obj,
                                                       comments_dp_obj=comments_dp_obj)
    # extracting LIWC features (before and while the experiment)
    pre_exper_info_exists = True if historical_sr_obj is not None else False
    while_exper_info_exists = True if cur_community_obj.comments_as_list is not None else False
    cur_community_obj.extract_liwc_features(pre_exper=pre_exper_info_exists, while_exper=while_exper_info_exists)
    # extracting the  meta & structural features that were created **during** the experiment
    cur_community_obj.extract_exper_meta_features()
    cur_community_obj.extract_exper_struct_features()

    # extracting textual data for BERT (the new function we wrote to be able to run BERT later) - while and a
    cur_community_obj.prepare_text_for_bert(submissions_sampling=bert_submissions_sampling,
                                            sampling_logic=bert_sampling_logic,
                                            sampling_percentage=bert_sampling_percentage, max_subm=bert_max_subm)
    # once we finish the whole object - we save it
    pickle.dump(cur_community_obj, open(saving_f_name, "wb"))
    if verbose:
        func_duration = (datetime.datetime.now() - func_start_time).seconds
        print(f"Thread idx {index} reports: Community object was built for {sr_name} in {func_duration} seconds.")
    return sr_name, sr_obj_found, struct_features_pre_exper_found


# the following main code is used to generate the full community objects - need to run it first before anything else
"""
if __name__ == "__main__":
    com_level_gold_label_df = \
        pickle.load(open(opj(data_path, 'success_analysis_target_df_09_02_2022.p'), "rb"))
    com_names = list(set(com_level_gold_label_df.index))
    com_names.sort()
    com_names = com_names[5:]
    # if required, we create users information (it is used by the graph classification algorithms). This should be done
    # only once (if it was already done and created - consider not running it again)
    if generate_users_info_data:
        users_info = create_users_data_dict(csv_data_path="/data/work/data/reddit_place/place_classifier_csvs",
                                            srs_to_pull_users_from=com_names, start_period='2017-01',
                                            end_period='2017-03')
        users_info_saving_path = os.path.join(os.path.dirname(sr_objects_path), 'users_data')
        pickle.dump(users_info, open(os.path.join(users_info_saving_path, "users_info_09_02_2022.p"), "wb"))
        duration = (datetime.datetime.now() - start_time).seconds
        print(f"\nFinished creating the whole users info file. File has been saved under {users_info_saving_path} "
              f"time up to now: {duration} seconds")

    # if required, we create subreddit objects (sr objects), which are later used by the algorithms
    if generate_sr_obj:
        # in case we wish NOT to overwrite existing objects - we should filter the existing ones
        if not overwrite_sr_obj:
            existing_sr_obj_names = [str(f.split('drawing_sr_obj_')[1]).split('.p')[0]
                                     for f in os.listdir(sr_obj_saving_path) if f.startswith('drawing_sr_obj')]
            unbuilt_com_names = [cn for cn in com_names if cn not in existing_sr_obj_names]
            data_for_poll = [(idx, cur_sr_name) for idx, cur_sr_name in enumerate(unbuilt_com_names)]
        else:
            data_for_poll = [(idx, cur_sr_name) for idx, cur_sr_name in enumerate(com_names)]
        duration = (datetime.datetime.now() - start_time).seconds
        print(f"Starting the pool process of ({len(data_for_poll)} instances). Time up to now: {duration} seconds.")
        pool = mp.Pool(processes=cpu_n)
        with pool as pool:
            results = pool.starmap(calc_and_save_community_obj, data_for_poll)
        unfound_explanatory_features = [name for name, found, _ in results if found == 0]
        unfound_graph_features = [name for name, _, found in results if found == 0]
        duration = (datetime.datetime.now() - start_time).seconds
        print(f"\nFinished creating the whole corpus. Duration: {duration} seconds.\n"
              f"For the following srs ({len(unfound_explanatory_features)}), no explanatory features were found: "
              f"{unfound_explanatory_features}.\n"
              f"For the following srs ({len(unfound_graph_features)}), no structural (SNA) features were found: "
              f"{unfound_graph_features}.\n"
              f"Objects are saved in: {os.path.join(data_path,'drawing_sr_objects')}")

"""
# the following main code is used to take the exsiting full community objects and create concise objects out of them
# it may run AFTER the previous main code
if __name__ == "__main__":
    existing_unconcised_f_names = [f for f in os.listdir(sr_obj_saving_path) if f.startswith('drawing_sr_obj')]
    existing_unconcised_community_names = [str(f.split('drawing_sr_obj_')[1]).split('.p')[0] for f in existing_unconcised_f_names]
    # checking which concise files has already been created
    existing_concise_f_names = [f for f in os.listdir(concise_sr_obj_saving_path) if f.startswith('concise_drawing_sr_obj')]
    existing_concise_community_names = [str(f.split('concise_drawing_sr_obj_')[1]).split('.p')[0]
                                        for f in existing_concise_f_names]
    communties_to_concise = [f for f in existing_unconcised_community_names if f not in existing_concise_community_names]
    # there should be 1307 in total which should be concised
    print(f"We plan to concise {len(communties_to_concise)} communities to concised file size.", flush=True)
    for cur_f_name in communties_to_concise:
        load_and_concise_community_obj(index=0, sr_name=cur_f_name, verbose=True)
    # grab all existing community names from the community_objects folder

