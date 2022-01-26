# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021


import warnings
warnings.simplefilter("ignore")
import gc
import sys
import os
import numpy as np
import collections
import datetime
import pickle
import multiprocessing as mp
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from r_place_drawing_classifier.utils import get_submissions_subset, get_comments_subset
from data_loaders.general_loader import sr_sample_based_subscribers, sr_sample_based_submissions
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
from sr_classifier.sub_reddit import SubReddit
import commentjson
import re
from pandas import Timestamp

STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

###################################################### Configurations ##################################################
config_dict = commentjson.load(open(os.path.join(os.getcwd(), 'config', 'srs_creation_config.json')))
machine = '' # name of the machine to be used. This should be sync with the config file
data_path = config_dict['data_dir'][machine]
batch_number = 0
########################################################################################################################


def _sr_creation(srs_mapping, submission_dp_obj, comments_dp_obj, srs_to_create, pool_number):
    print("Pool # {} has started running the _sr_creation function".format(pool_number))
    start_time = datetime.datetime.now()
    empty_srs = 0
    submission_data = get_submissions_subset(
        files_path=os.path.join(data_path, 'place_classifier_csvs'), srs_to_include=srs_to_create,
        start_month=config_dict['data_period']['start_month'], end_month=config_dict['data_period']['end_month'],
        min_utc=None, max_utc='2017-03-29 00:00:00')

    # same thing for the comments data
    if eval(config_dict['comments_usage']['meta_data']) or eval(config_dict['comments_usage']['corpus']):
        comments_data = get_comments_subset(files_path=os.path.join(data_path, 'place_classifier_csvs'),
                                            srs_to_include=srs_to_create,
                                            start_month=config_dict['data_period']['start_month'],
                                            end_month=config_dict['data_period']['end_month'],
                                            min_utc=None, max_utc='2017-03-29 00:00:00')
    else:
        comments_data = None

    for idx, sr_name in enumerate(srs_to_create):
        cur_sr_submission = submission_data[submission_data['subreddit'].str.lower() == sr_name]
        # case there are no relevant submissions to this sr
        if cur_sr_submission.shape[0] == 0:
            empty_srs += 1
            continue
        # pulling out the meta data about the SR (the iloc[0] is used only to convert it into Series)
        sr_meta_data = {'SR': sr_name, 'creation_utc': srs_mapping[sr_name][1], 'end_state': None,
                        'num_of_users': srs_mapping[sr_name][0],
                        'trying_to_draw': 'Yes' if srs_mapping[sr_name][2] == 'drawing' else 'No', 'models_prediction': -1}
        cur_sr_obj = SubReddit(meta_data=sr_meta_data)
        cur_sr_submission_after_dp, submission_text = submission_dp_obj.data_pre_process(reddit_df=cur_sr_submission)
        # updating the object with the list of submissions
        cur_sr_obj.submissions_as_list = submission_text
        # detecting language of the text
        subm_to_take = min(len(cur_sr_obj.submissions_as_list), 1000)
        data_for_lang_detector = [sal[1] + '. ' + sal[2] for sal in cur_sr_obj.submissions_as_list[0:subm_to_take]
                                  if type(sal[2]) is str and type(sal[1]) is str]
        chosen_lang = submission_dp_obj.detect_lang(text_list=data_for_lang_detector, min_score_per_sent=0.9,
                                                    min_agg_score=0.7, sr_name=cur_sr_obj.name, verbose=False)
        cur_sr_obj.lang = chosen_lang
        del cur_sr_submission
        del data_for_lang_detector
        gc.collect()
        # case we wish to tokenize the submission data, we'll do it now
        full_tok_text = []
        for s in submission_text:
            # case the self-text is not none (in case it is none, we'll just take the header or the self text)
            if type(s[2]) is str and type(s[1]) is str:
                sample_for_tokenizer = submission_dp_obj.mark_urls(s[1], marking_method='tag')[0] + '. ' + \
                                       submission_dp_obj.mark_urls(s[2], marking_method='tag')[0]
                cur_tok_words = submission_dp_obj.tokenize_text(sample_for_tokenizer, convert_to_lemmas=False,
                                                                break_to_sents=True)
            elif type(s[1]) is str:
                sample_for_tokenizer = submission_dp_obj.mark_urls(s[1], marking_method='tag')[0]
                cur_tok_words = submission_dp_obj.tokenize_text(sample=sample_for_tokenizer, convert_to_lemmas=False,
                                                                break_to_sents=True)
            elif type(s[2]) is str:
                sample_for_tokenizer = submission_dp_obj.mark_urls(s[2], marking_method='tag')[0]
                cur_tok_words = submission_dp_obj.tokenize_text(sample=sample_for_tokenizer, convert_to_lemmas=False,
                                                                break_to_sents=True)
            else:
                continue
            full_tok_text.append(cur_tok_words)
        cur_sr_obj.submissions_as_tokens = full_tok_text
        del full_tok_text

        # pulling out the comments data - case we want to use it. There are a few option of comments usage
        # case we want to use comments data for either creation of meta-data and as part of the corpus (or both)
        if eval(config_dict['comments_usage']['meta_data']) or eval(config_dict['comments_usage']['corpus']):
            # first, we filter the comments, so only ones in the current sr we work with will appear
            cur_sr_comments = comments_data[comments_data['subreddit'].str.lower() == sr_name]
            submission_ids = set(['t3_' + sub_id for sub_id in cur_sr_submission_after_dp['id']])
            # second, we filter the comments, so only ones which are relevant to the submissions dataset will appear
            # (this is due to the fact that we have already filtered a lot of submissions in pre step)
            cur_sr_comments = cur_sr_comments[cur_sr_comments['link_id'].isin(submission_ids)]
            cur_sr_comments_after_dp, comments_text = comments_dp_obj.data_pre_process(reddit_df=cur_sr_comments)
            del cur_sr_comments
        # case we want to use comments data for meta-features creation (very logical to be used)
        if eval(config_dict['comments_usage']['meta_data']):
            cur_sr_obj.create_explanatory_features(submission_data=cur_sr_submission_after_dp,
                                                   comments_data=cur_sr_comments_after_dp)
        # case we want to use only submission data for meta-data creation
        else:
            cur_sr_obj.create_explanatory_features(submission_data=cur_sr_submission_after_dp, comments_data=None)
        # case we want to use comments data as part of the corpus creation (most of the times not the case)
        if eval(config_dict['comments_usage']['corpus']):
            cur_sr_obj.comments_as_list = comments_text

            # case we wish to tokenize the comments data, we'll do it now
            full_tok_text = []
            for s in comments_text:
                if type(s[1]) is str:
                    sample_for_tokenizer = comments_dp_obj.mark_urls(s[1], marking_method='tag')[0]
                    cur_tok_words = comments_dp_obj.tokenize_text(sample=sample_for_tokenizer, convert_to_lemmas=False,
                                                                  break_to_sents=True)
                    full_tok_text.append(cur_tok_words)
            cur_sr_obj.comments_as_tokens = full_tok_text
            del full_tok_text
        # updating the object with dictionaries of both submissions and comments
        cur_sr_obj.update_words_dicts(update_only_submissions_dict=False)
        # saving a pickle file of the object
        file_name = \
            os.path.join(data_path, #'sr_objects',
                         str('sr_obj_' + sr_name + '_' + config_dict['saving_options']['file_name_suffix'] + '.p'))
        # case the file name exists, we will raise a warning about it and will replace it
        if os.path.exists(file_name) and eval(config_dict['saving_options']['override_existing_files']) and eval(config_dict['saving_options']['save_obj']):
            warnings.warn("sr_obj file for sr {} was found, will be replaced by a new one".format(sr_name))
            pickle.dump(cur_sr_obj, open(file_name, "wb"))
        elif not os.path.exists(file_name) and eval(config_dict['saving_options']['save_obj']):
            pickle.dump(cur_sr_obj, open(file_name, "ab"))

        gc.collect()
        # printing progress
        if idx % 1 == 0 and idx != 0:
            duration = (datetime.datetime.now() - start_time).seconds
            print("Pool # {} reporting finish of the {} iteration. Took this pool {} seconds, "
                  "moving forward".format(pool_number, idx, duration))
    duration = (datetime.datetime.now() - start_time).seconds
    print("Pool # {} reporting finished passing over all SRs. Took him up to now {} seconds and he created {} SRs "
          "objects (some where empty, so weren't created)".format(pool_number, duration, len(srs_to_create)))
    del submission_data
    del comments_data
    gc.collect()
    return 0
    # return sr_objects


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # first step will be to sample the data, then we will represent the result as a dictionary here
    if config_dict['under_sample']['logic'] == 'subscribers':
        full_srs_mapping = sr_sample_based_subscribers(data_path=data_path, sample_size='1:1',
                                                       threshold_to_define_as_drawing=0.7,
                                                       balanced_sampling_based_sr_size=True, internal_sr_metadata=True,
                                                       sr_statistics_usage=True)

    # sampling SRs based on number of submissions
    elif config_dict['under_sample']['logic'] == 'submissions':
        under_sam_config = config_dict['under_sample']
        full_srs_mapping = sr_sample_based_submissions(data_path=data_path, sample_size=under_sam_config['ratio'],
                                                       start_period=under_sam_config['start_month'],
                                                       end_period=under_sam_config['end_month'],
                                                       threshold_to_define_as_drawing=under_sam_config['drawing_threshold'])
    else:
        raise IOError("Under-Sampling method must be either 'submissions' or 'subscribers'. Fix and try again")
    full_srs_mapping = {sr[0]: (sr[1], sr[2], sr[3]) for sr in full_srs_mapping}
    full_srs_names = list(full_srs_mapping.keys())
    # sorting the names list, so we can transfer it to couple of slaves on the cluster without worrying about the order
    full_srs_names.sort()
    # filtering the names to only the current batch we wish to use
    srs_names = full_srs_names[batch_number*200:(batch_number+1)*200]
    srs_mapping = {key: value for key, value in full_srs_mapping.items() if key in srs_names}
    srs_mapping_len = len(srs_mapping)
    # case some of the objects were created, and we don't wish to re-create them , we'll remove them from the list
    if not eval(config_dict["saving_options"]["override_existing_files"]):
        sr_files_path = os.path.join(data_path, 'sr_objects')
        files_suffix = config_dict['saving_options']['file_name_suffix'] + '_.p'
        existing_files = [f for f in os.listdir(sr_files_path) if re.match(r'sr_obj_.*\_\.p', f)]
        existing_srs = {re.search('sr_obj_(.*)'+files_suffix, fn).group(1) for fn in existing_files}
        srs_mapping = {key: value for key, value in srs_mapping.items() if key not in existing_srs}
    srs_names = list(srs_mapping.keys())
    # sorting the names list, so we can transfer it to couple of slaves on the cluster without worrying about the order
    srs_names.sort()
    srs_skipped_len = srs_mapping_len - len(srs_mapping)
    srs_mapping_len = len(srs_mapping)
    print("We are going to handle {} srs. We did not take into account {} srs, since we found their pickle file."
          "This is the drawing/not-drawing "
          "distribution: {}".format(srs_mapping_len, srs_skipped_len,
                                    collections.Counter([value[2] for key, value in srs_mapping.items()])))

    if srs_mapping_len > 0:
        # pulling the submission data, based on the subset of SRs we decided on
        # Data prep - train
        submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
        comments_dp_obj = RedditDataPrep(is_submission_data=False, remove_stop_words=False, most_have_regex=None)
        processes_amount = 1
        chunk_size = int(len(srs_names) * 1.0 / processes_amount)
        srs_names_in_chunks = [srs_names[i * chunk_size: i * chunk_size + chunk_size] for i in
                               range(processes_amount - 1)]
        # last one can be tricky, so handling it more carefully
        srs_names_in_chunks.append(srs_names[(processes_amount - 1) * chunk_size:])
        input_for_pool = [(srs_mapping, submission_dp_obj, comments_dp_obj, srs_names_in_chunks[i], i) for i in
                          range(processes_amount)]
        pool = mp.Pool(processes=processes_amount)
        with pool as pool:
            results = pool.starmap(_sr_creation, input_for_pool)        
    duration = (datetime.datetime.now() - start_time).seconds
    print("\nTotal run time is: {}".format(duration))