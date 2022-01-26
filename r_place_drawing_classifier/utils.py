# Authors: Avrahami Israeli (isabrah)
# Python version: 3.7
# Last update: 04.08.2019

import datetime
import pandas as pd
import re
import os
import collections
import pickle
import sys
import csv
import random


def get_submissions_subset(files_path, srs_to_include, start_month='2016-10', end_month='2017-03',
                           min_utc=None, max_utc='2017-03-29 00:00:00'):
    """
    pulling our subset of the submission data, which is related to the list of SRs given as input. This is very
    simple "where" statement
    :param files_path: string
        location of the files to be used (.csv ones)
    :param srs_to_include: list (maybe set will also work here)
        list with SR names to be included in the returned dataset. Expected to be lowe-case ones.
        If None is given as input, then ALL subrddits are included
    :param start_month: string, default: '2016-10'
        the starting month in YYYY-MM format which data should be taken from
    :param end_month: string, default: '2017-03'
        the ending month in YYYY-MM format which data should be taken from
    :param min_utc: string, default: None
        the minimum timestamp to take into account. If None - no minimum time limitation is taken into account
    :param max_utc: string, default: '2017-03-29 00:00:00' (a day before the start time of r/place experiment)
        the maximum timestamp to take into account. If None - no minimum time limitation is taken into account
    :return: pandas data-frame
        df including all relevant submission, related to the SRs given as input
    """
    start_time = datetime.datetime.now()
    # finding all the relevant zip files in the 'data_path' directory
    submission_files = [f for f in os.listdir(files_path) if re.match(r'RS.*\.csv', f) and 'sample' not in f]
    # taking only the submissions files from 10-2016 to 03-2017
    submission_files = [i for i in submission_files if
                        ''.join(['RS_', start_month, '.csv']) <= i <= ''.join(['RS_', end_month, '.csv'])]
    submission_files = sorted(submission_files)
    submission_dfs = []
    # iterating over each submission file
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=os.path.join(files_path, cur_submission_file), encoding='utf-8')
        # filtering the data-frame based on the list of SRs we want to include and the date (before r/place started)
        if srs_to_include is not None:
            cur_submission_df = cur_submission_df[cur_submission_df["subreddit"].str.lower().isin(srs_to_include)]
        # filtering based on min/max date
        cur_min_utc = min(cur_submission_df['created_utc_as_date']) if min_utc is None else min_utc
        cur_max_utc = max(cur_submission_df['created_utc_as_date']) if max_utc is None else max_utc
        cur_submission_df = cur_submission_df[(cur_submission_df['created_utc_as_date'] >= cur_min_utc) &
                                              (cur_submission_df['created_utc_as_date'] <= cur_max_utc)]
        submission_dfs.append(cur_submission_df)
    if len(submission_dfs) == 0:
        raise IOError("No submission file was found")

    full_submissions_df = pd.concat(submission_dfs)
    duration = (datetime.datetime.now() - start_time).seconds
    print(f"Function 'get_submission_subset_dataset' has ended. Took us : {duration} seconds. "
          f"Submission data-frame shape created is {full_submissions_df.shape}", flush=True)
    return full_submissions_df


def get_comments_subset(files_path, srs_to_include, start_month='2016-10', end_month='2017-03',
                        min_utc=None, max_utc='2017-03-29 00:00:00'):
    """
    pulling our subset of the commnets data, which is related to the list of SRs given as input. This is very
    simple "where" statement
    :param files_path: string
        location of the files to be used (.csv ones)
    :param srs_to_include: list (maybe set will also work here)
        list with SR names to be included in the returned dataset. Expected to be lowe-case ones
        If None is given as input, then ALL subrddits are included
    :param start_month: string
        the starting month in YYYY-MM format which data should be taken from
    :param end_month: string
        the ending month in YYYY-MM format which data should be taken from
    :param min_utc: string, default: None
        the minimum timestamp to take into account. If None - no minimum time limitation is taken into account
    :param max_utc: string, default: '2017-03-29 00:00:00' (a day before the start time of r/place experiment)
        the maximum timestamp to take into account. If None - no minimum time limitation is taken into account
    :return: pandas data-frame
        df including all relevant submission, related to the SRs given as input
    """
    start_time = datetime.datetime.now()
    # pulling out all comment files in the desired range of months
    comments_files = [f for f in os.listdir(files_path) if re.match(r'RC.*\.csv', f) and 'sample' not in f]
    comments_files = [i for i in comments_files if
                      ''.join(['RC_', start_month, '.csv']) <= i <= ''.join(['RC_', end_month, '.csv'])]
    comments_files = sorted(comments_files)
    comments_dfs = []
    # looping over each file
    for comm_idx, cur_comments_file in enumerate(comments_files):
        if sys.platform == 'linux':
            cur_comments_df = pd.read_csv(filepath_or_buffer=os.path.join(files_path, cur_comments_file), encoding='latin-1')
        else:
            cur_comments_df = pd.read_csv(filepath_or_buffer=os.path.join(files_path, cur_comments_file), encoding='latin-1')
        if srs_to_include is not None:
            cur_comments_df = cur_comments_df[cur_comments_df["subreddit"].str.lower().isin(srs_to_include)]
        # filtering based on min/max date
        cur_min_utc = min(cur_comments_df['created_utc_as_date']) if min_utc is None else min_utc
        cur_max_utc = max(cur_comments_df['created_utc_as_date']) if max_utc is None else max_utc
        cur_comments_df = cur_comments_df[(cur_comments_df['created_utc_as_date'] >= cur_min_utc) &
                                          (cur_comments_df['created_utc_as_date'] <= cur_max_utc)]
        comments_dfs.append(cur_comments_df)
    if len(comments_dfs) == 0:
        raise IOError("No comments file were found")
    full_comments_df = pd.concat(comments_dfs)
    duration = (datetime.datetime.now() - start_time).seconds
    print(f"Function 'get_comments_subset' has ended. Took us : {duration} seconds. "
          f"Comments data-frame shape created is {full_comments_df.shape}", flush=True)
    return full_comments_df


def calc_sr_statistics(files_path, included_years, saving_res_path=os.getcwd()):
    """
    calculating relevant statistics to each sr found in the files given as input. This will be later used in order
    to filter SRs with no submission/very small amount of submissions
    :param files_path: string
        location of the files to be used (.csv ones)
    :param included_years: list
        list of years to include in the analysis
    :param saving_res_path: string
        location where to save results
    :return: dictionary
        dictionary with statistics to each SR. The function also saves the results as a pickle file

    Example:
    --------
    >>> data_path = "/home/isabrah/reddit_pycharm_proj_with_own_pc/data_loaders/r_place_classification/place_classifier_csvs/" if sys.platform == "linux" else "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\place_classifier_csvs\\"
    >>> dict_res = calc_sr_statistics(files_path=data_path, saving_res_path=data_path, included_years=[2017, 2016], submission_only=True)
    """
    start_time = datetime.datetime.now()
    submission_files = [f for f in os.listdir(files_path) if re.match(r'RS.*\.csv', f)]
    # taking only files which are in the 'included_years' subset
    submission_files = [sf for sf in submission_files if any(str(year) in sf for year in included_years)]
    # comments_files = [cf for cf in comments_files if any(str(year) in cf for year in included_years)]
    submission_files = sorted(submission_files)
    sr_statistics = collections.Counter()
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=os.path.join(files_path,cur_submission_file))
        cur_sr_statistics = collections.Counter(cur_submission_df["subreddit"])
        sr_statistics += cur_sr_statistics
        # writing status to screen
        duration = (datetime.datetime.now() - start_time).seconds
        print("Ended loop # {}, up to now took us {} seconds".format(subm_idx, duration))
    # saving the stats to a file
    pickle.dump(sr_statistics, open(saving_res_path + "submission_stats_102016_to_032017.p", "wb"))
    duration = (datetime.datetime.now() - start_time).seconds
    print("Function 'calc_sr_statistics' has ended. Took us : {} seconds. "
          "Final dictionary size is {}".format(duration, len(sr_statistics)))
    return sr_statistics


def save_results_to_csv(results_file, start_time, objects_amount, config_dict, results):
    """
    given inputs regarding a final run results - write these results into a file
    :param results_file: str
        file of the csv where results should be placed
    :param start_time: datetime
        time when the current result run started
    :param objects_amount: int
        amount of objects in the run was based on, usually it is between 1000-2500
    :param config_dict: dict
        dictionary holding all the configuration of the run, the one we get as input json
    :param results: dict
        dictionary with all results. Currently it should contain the following keys: 'accuracy', 'precision', 'recall'
    :return: None
        Nothing is returned, only saving to the file is being done
    """

    file_exists = os.path.isfile(results_file)
    rf = open(results_file, 'a', newline='')
    with rf as output_file:
        dict_writer = csv.DictWriter(output_file,
                                     fieldnames=['timestamp', 'start_time', 'machine', 'SRs_amount', 'cv_folds',
                                                 'configurations', 'accuracy', 'precision', 'recall', 'auc'])
        # only in case the file doesn't exist, we'll add a header
        if not file_exists:
            dict_writer.writeheader()
        try:
            host_name = os.environ['HOSTNAME'] if sys.platform == 'linux' else os.environ['COMPUTERNAME']
        except KeyError:
            host_name = 'pycharm with this ssh: ' + os.environ['SSH_CONNECTION']
        dict_writer.writerow({'timestamp': datetime.datetime.now(), 'start_time': start_time,
                              'SRs_amount': objects_amount, 'machine': host_name, 'cv_folds': len(results['accuracy']),
                              'configurations': config_dict, 'accuracy': results['accuracy'],
                              'precision': results['precision'], 'recall': results['recall'], 'auc': results['auc']})
    rf.close()


def examine_word(sr_object, regex_required, tokenizer, saving_file=os.getcwd()):
    """
    analysis function to see where a specific word is being used in the submissions corpus. This is useful in order to
    see how different communities/users use different words in Reddit

    :param sr_object: SubReddit object
        an object (which can be seen in sr_classifier.sub_reddit) containing information about a community
    :param regex_required: str
        the regex we want to examine - can be a single word/bi-gram or any other regex
    :param tokenizer: tokenizer object
        function which can be used to tokenize the data
        Example how it can be defined:
        >>>submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
        >>>reddit_tokenizer = submission_dp_obj.tokenize_text
    :param saving_file: str
        the full path to the file (including its name) where results should be saved in
    :return: nothing
    """
    start_time = datetime.datetime.now()
    print("examine_word function has started")
    tot_cnt = 0
    if sys.platform == 'linux':
        explicit_file_name = saving_file + '/' + 'examine_word_res_regex_' + regex_required + '.txt'
    else:
        explicit_file_name = saving_file + '\\' + 'examine_word_res_regex_' + regex_required + '.txt'
    if os.path.exists(explicit_file_name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(explicit_file_name, append_write, encoding="utf-8") as text_file:
        text_file.write("\n\nHere are the relevant sentences with the regex {} in SR named {}. This sr is labeled as: "
                        "{}".format(regex_required, sr_object.name,
                                    'not drawing' if sr_object.trying_to_draw == -1 else 'drawing'))

        for subm in sr_object.submissions_as_list:
            normalized_text = []
            try:
                tokenized_txt_title = tokenizer(subm[1])
                normalized_text += tokenized_txt_title
            except TypeError:
                pass
            try:
                tokenized_txt_selftext = tokenizer(subm[2])
                normalized_text += tokenized_txt_selftext
            except TypeError:
                pass
            if regex_required in set(normalized_text):
                text_file.write('\n' + str(subm))
                tot_cnt += 1
        duration = (datetime.datetime.now() - start_time).seconds
        print("examine_word has ended, took us {} seconds."
              "Total of {} rows were written to a text file".format(duration, tot_cnt))


def remove_huge_srs(sr_objects, quantile=0.01):
    """
    removed the largest sr objects - in order not to handle big srs with lots of submissions/comments
    :param sr_objects: list
        list of sr objects
    :param quantile: float, default=0.01
        the % of srs required to remove
    :return: list
        the list of srs after removing the huge ones from it
    """
    srs_summary = [(idx, cur_sr.name, cur_sr.trying_to_draw, len(cur_sr.submissions_as_list))
                   for idx, cur_sr in enumerate(sr_objects)]
    srs_summary.sort(key=lambda tup: tup[3], reverse=True)  # sorts in place according to the # of submissions
    amount_of_srs_to_remove = int(len(sr_objects)*quantile)
    srs_to_remove_summary = srs_summary[0:amount_of_srs_to_remove]
    drawing_removed_srs = sum([sr[2] for sr in srs_to_remove_summary if sr[2]==1])
    srs_to_remove = set([sr[1] for sr in srs_to_remove_summary])
    returned_list = [sr for sr in sr_objects if sr.name not in srs_to_remove]
    print("remove_huge_srs function has ended, {} srs have been removed,"
          " {} out of them are from class 1 (drawing)".format(len(srs_to_remove), drawing_removed_srs))
    return returned_list


def check_input_validity(config_dict, machine):
    """
    Set of tests to make sure the input configuration we have is valid.
    If not - we either fix it or change it to be valid
    :param config_dict: dict
        the configuration dictionary input. This is read as a json in the main file
    :param machine: str
        the machine (computer) we work on (e.g. 'yalla')
    :return: dict
        the updated config_dict (dictionary) or an error
    """
    # making sure the model we get as input is valid
    if config_dict['class_model']['model_type'] not in ['clf_meta_only', 'bow', 'mlp',
                                                        'single_lstm', 'parallel_lstm', 'cnn_max_pooling']:
        raise IOError('Model name input is invalid. Must be one out of the following: '
                      '["clf_meta_only", "bow", "mlp", "single_lstm", "parallel_lstm", "cnn_max_pooling]. '
                      'Please fix and try again')

    # checking if a folder with the model name exists already - if it exists, we will change name of the model
    cur_folder_name = os.path.join(config_dict['results_dir'][machine], "model_" + config_dict['model_version'])
    if os.path.exists(cur_folder_name):
        new_model_version = config_dict['model_version'] + '.' + str(random.randint(1, 100000000))
        config_dict['model_version'] = new_model_version
        print("model_version has been changed to {}, "
              "since the original model_version was used in the past".format(new_model_version))
    return config_dict
