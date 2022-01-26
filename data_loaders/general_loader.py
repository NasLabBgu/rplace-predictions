# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 6.11.2018

import datetime
import pandas as pd
import numpy as np
import re
import sys
import os
import bz2
import json
import csv
import lzma
import pickle
import collections
from collections import OrderedDict, defaultdict

###################################################### Configurations ##################################################
data_path = '/data/work/data/reddit_general/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
data_to_process = 'both'   # can be either 'submission' / 'comments' / 'both'
included_years = [2017]

submission_columns = ['author_flair_css_class','saved', 'media', 'score', 'distinguished', 'from_id', 'title',
                      'num_comments', 'subreddit_id', 'link_flair_text', 'link_flair_css_class', 'stickied',
                      'created_utc', 'thumbnail', 'ups', 'retrieved_on', 'locked', 'gilded', 'secure_media_embed',
                      'from', 'from_kind', 'secure_media', 'is_self', 'hide_score', 'permalink', 'over_18', 'domain',
                      'subreddit', 'author_flair_text', 'selftext', 'url', 'name', 'archived', 'downs', 'quarantine',
                      'id', 'edited', 'media_embed', 'author', 'created_utc_as_date']
comments_columns = ['created_utc', 'link_id', 'subreddit_type', 'controversiality', 'author', 'id', 'body',
                    'retrieved_on', 'author_flair_text', 'can_gild', 'distinguished', 'subreddit_id', 'score',
                    'subreddit', 'gilded', 'is_submitter', 'permalink', 'stickied', 'author_flair_css_class', 'edited',
                    'parent_id', 'created_utc_as_date']
########################################################################################################################


def general_loader(data_path, sr_to_include=None, saving_path=os.getcwd(), load_only_columns_subset=False):
    """
    loading reddit data, based on the zipped files from here - https://files.pushshift.io/reddit/
    the process converts this files into csv, after filtering some columns (if load_only_columns_subset=True) and
    adding a date columns

    :param data_path: str
        location of the data
    :param sr_to_include: list or None, default: NOne
        placeholder for future use - to be used when only subset of SRs are needed
    :param saving_path: str, default: existing location of the python code
        location to save the files into
    :param load_only_columns_subset: bool, default: False
    :return: None
        only prints to screen and saving csv files into the saving_path location

    Example
    -------
    >>> general_loader(data_path, saving_path=data_path + 'place_classifier_csvs', load_only_columns_subset=True)
    """

    start_time = datetime.datetime.now()
    # finding all the relevant zip files in the 'data_path' directory
    submission_files_path = data_path + 'submissions/' if sys.platform == 'linux' else data_path + 'submissions\\'
    comments_files_path = data_path + 'comments/' if sys.platform == 'linux' else data_path + 'comments\\'
    submission_files = [f for f in os.listdir(submission_files_path) if re.match(r'RS.*\.bz2|RS.*\.xz', f)]
    comments_files = [f for f in os.listdir(comments_files_path) if re.match(r'RC.*\.bz2|RC.*\.xz', f)]
    # taking only files which are in the 'included_years' subset
    submission_files = [i for i in submission_files if 'RS_2017-04.bz2' <= i <= 'RS_2017-04.bz2']
    comments_files = [i for i in comments_files if 'RC_2017-04.bz2' <= i <= 'RC_2017-04.bz2']
    submission_files = sorted(submission_files)
    comments_files = sorted(comments_files)
    submissions_interesting_col = ["created_utc_as_date", "author", "subreddit", "title", "selftext", "num_comments",
                                   "permalink", "score", "id", "thumbnail"]
    comments_interesting_col = ["created_utc_as_date", "author", "subreddit", "body", "score", "id",
                                "link_id", "parent_id", "thumbnail"]
    # looping over each file of the submission/comment and handling it
    if data_to_process == 'submission' or data_to_process == 'both':
        # for cur_submission_file in submission_files[0]:
        for subm_idx, cur_submission_file in enumerate(submission_files):
            if cur_submission_file.endswith('bz2'):
                zipped_submission = bz2.BZ2File(submission_files_path + cur_submission_file, 'r')
            else:
                zipped_submission = lzma.open(submission_files_path + cur_submission_file, mode='r')
            # looping over each row in the submission data
            submissions = []
            for inner_idx, line in enumerate(zipped_submission):
                try:
                    cur_line = json.loads(line.decode('UTF-8'))
                except json.decoder.JSONDecodeError:
                    continue
                cur_line['created_utc_as_date'] = str(pd.to_datetime(cur_line['created_utc'], unit='s'))
                if load_only_columns_subset:
                    line_shrinked = dict((k, cur_line[k]) if k in cur_line else (k, None) for k in submissions_interesting_col)
                # we still define this 'line_shrinked' also in cases when we want to have all columns, since in some
                # cases there are redundant columns appear in the zip original files
                else:
                    line_shrinked = dict((k, cur_line[k]) if k in cur_line else (k, None) for k in submission_columns)
                submissions.append(line_shrinked)
                #if inner_idx > 100000:
                #    break
            # saving the file to disk. Currently it is as a csv format (found it as the most useful one)
            full_file_name = saving_path + '/' if sys.platform == 'linux' else saving_path + '\\'
            f = open(full_file_name + cur_submission_file[:-4] + '.csv', mode='a', encoding="utf-8")
            keys = submissions[0].keys()
            with f as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(submissions)
            duration = (datetime.datetime.now() - start_time).seconds
            print("Finished handling the {}'th submission file called {}. Took us up to now: {} seconds. "
                  "Current submission size is {}".format(subm_idx + 1, cur_submission_file, duration, len(submissions)))
    # handling the comments data - exactly same logic as the submission data handling
    if data_to_process == 'comments' or data_to_process == 'both':
        for comm_idx, cur_comments_file in enumerate(comments_files):
            if cur_comments_file.endswith('bz2'):
                zipped_comments = bz2.BZ2File(comments_files_path + cur_comments_file, 'r')
            else:
                zipped_comments = lzma.open(comments_files_path + cur_comments_file, mode='r')
            # looping over each row in the comments data
            comments = []
            for inner_idx, line in enumerate(zipped_comments):
                try:
                    cur_line = json.loads(line.decode('UTF-8'))
                except json.decoder.JSONDecodeError:
                    continue
                cur_line['created_utc_as_date'] = pd.to_datetime(cur_line['created_utc'], unit='s')
                if load_only_columns_subset:
                    line_shrinked = dict((k, cur_line[k]) for k in comments_interesting_col if k in cur_line)
                # we still define this 'line_shrinked' also in cases when we want to have all columns, since in some
                # cases there are redundant columns appear in the zip original files
                else:
                    line_shrinked = dict((k, cur_line[k]) if k in cur_line else (k, None) for k in comments_columns)
                comments.append(line_shrinked)
                #if inner_idx > 100000:
                #    break
            # saving the file to disk
            full_file_name = saving_path + '/' if sys.platform == 'linux' else saving_path + '\\'
            f = open(full_file_name + cur_comments_file[:-4] + '.csv', mode='a', encoding="utf-8")
            keys = comments[0].keys()
            with f as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(comments)
            duration = (datetime.datetime.now() - start_time).seconds
            print("Finished handling the {}'th comments file. Took us up to now: {} seconds. "
                  "Current comments size is {}".format(comm_idx+1, duration, len(comments)))


def sr_sample_based_subscribers(data_path, sample_size, threshold_to_define_as_drawing, internal_sr_metadata=True,
                               sr_statistics_usage=True, balanced_sampling_based_sr_size=False, seed=1984):
    """
    sampling group of SRs to be later used for modeling. This group is in most cases the 'not-drawing' teams. We will
    use meta-data for sampling purposes, either internal meta-data or external one
    :param data_path: str
        location of all data needed for the algorithm
    :param sample_size: int or string of float
        how many SRs should be sampled. Can be represents as simple in (then, it should be in
        most cases >= len(not_drawing_SRs)). Can be float, and then it represents the % of SRs to choose. Can be string
        in the form of 'a:b' and then it represnts ratio between drawing and not-drawing team (e.g '2:1' means we will
        have not-drawing teams X2 compared to the drawing teams. '1:1' means pure balance situation)
    :param threshold_to_define_as_drawing: float
        a value to be used to define a SR as drawing, according to the value in the excel file. The value in the excel
        file represents our probability that a certain SR was drawing
    :param sr_statistics_usage: bool, default: True
        whether to use statisics we computer regarding SRs and thier uasge in reddit in the past. It allows us not to
        sample SRs which didn't submit any submission along the last 6 months
    :param internal_sr_metadata: bool, default: True
        whether to use internal or extrnal meta-data as supportive source of information.
        Internal is the summary of SRs mata-data we created, which is based on the submissions data and reddit API (can
        be seen in the crawl_srs_metadata.py file). External is the data coming from here -
        https://files.pushshift.io/reddit/subreddits/ (which can be not updated and missing some SRs we need)
    :param balanced_sampling_based_sr_size: bool, default: False
         whether or not to apply an algorithm which will do "smart" sampling and not random. It will balance the
         SRs between drawing/not-drawing in terms of SR size (so distribution of the two will be almost similar)
    :param sr_metadata_usage: bool, default: True
        whether or not to use meta-data for sampling purposes. This can indeed lead to better sampling (e.g., SRs
        without any submission data will not be chosen)
    :param seed: int, default: 1984
        the seed to be used for randomization
    :return: list of tuples
        list where each item is a SRs and contains some information about it (represented in a tuple):
        [0] contains its name, [1] contains num_of_users, [2] contains 'creation_utc', [3] contains string ('drawing'
        or 'not_drawing')
    """
    # in case we wish to use the data we crawled from reddit with explicit information about SRs writing along 2016-2017
    if internal_sr_metadata:
        srs_meta_data = defaultdict(list)
        with open(data_path + 'srs_meta_data_102016_to_032017.json') as f:
            for idx, line in enumerate(f):
                cur_line = json.loads(line)
                cur_sr_name = str(cur_line['display_name']).lower()
                # case we already 'met' this sr, we'll take the max out of all subscribers amount we see
                if cur_sr_name in srs_meta_data:
                    srs_meta_data[cur_sr_name] = [max(cur_line['subscribers'], srs_meta_data[cur_sr_name][0]),
                                                  cur_line['created']]

                elif cur_line['subscribers'] is not None:
                    srs_meta_data[cur_sr_name] = [cur_line['subscribers'], cur_line['created']]
        sr_basic = pd.DataFrame.from_dict(data=srs_meta_data, orient='index')
        sr_basic.reset_index(inplace=True)
        sr_basic.columns = ['subreddit_name', 'number_of_subscribers', 'creation_epoch']

    # if we don't want to use the metadata information we crawled, we will use other source of information related to
    # the meta-data of subreddits (coming from https://files.pushshift.io/reddit/subreddits/)
    else:
        # this is the big file, including ALL the SRs exists
        sr_basic = pd.read_csv(filepath_or_buffer=data_path+'subreddits/subreddits_basic.csv' if sys.platform == 'linux'
                                    else data_path + 'subreddits_basic.csv',
                               header=0, index_col=False)
        # removing from sr_basic rows we cannot handle (None, nulls, 'None') and
        # converting an important columns to numeric
        sr_basic = sr_basic[~((sr_basic['number_of_subscribers'].isnull()) | (sr_basic['number_of_subscribers'].isna()))]
        sr_basic = sr_basic.loc[sr_basic['number_of_subscribers'] != 'None']
    sr_basic['number_of_subscribers'] = pd.to_numeric(sr_basic['number_of_subscribers'])

    # adding explicit timestamp to the data and filtering SRs which were created before r/place started
    sr_basic = sr_basic.assign(created_utc_as_date=pd.to_datetime(sr_basic['creation_epoch'], unit='s'))
    sr_basic = sr_basic[sr_basic['created_utc_as_date'] < '2017-03-31 00:00:00']
    sr_basic.sort_values(by='created_utc_as_date', inplace=True)
    # this is the excel file we created, including only SRs related in some way to the r/place experiment
    place_related_srs = pd.read_excel(io=data_path + 'subreddits_revealed/'
                                                     'all_subredditts_based_atlas_and_submissions.xlsx'
                                      if sys.platform == 'linux' else data_path + 'sr_relations\\all_subredditts_based'
                                                                                  '_atlas_and_submissions.xlsx',
                                      sheet_name='Detailed list')
    drawing_srs = place_related_srs[(place_related_srs['trying_to_draw'] == 'Yes') |
                                    (place_related_srs['models_prediction'] > threshold_to_define_as_drawing)]['SR']
    # flag which determines whether we filter SRs based on the fact no submission was done by them in the last 6 months
    if sr_statistics_usage:
        submission_stats = pickle.load(open(data_path +
                                            "place_classifier_csvs/submission_stats_102016_to_032017.p", "rb"))
        active_srs = set(str(key).lower() for key in submission_stats.keys())
        drawing_srs = set([str(name).lower() for name in drawing_srs if str(name).lower() in active_srs])
        sr_basic_names = set([str(name).lower() for name in sr_basic['subreddit_name'] if str(name).lower() in active_srs])
    else:
        drawing_srs = set([str(name).lower() for name in drawing_srs])
        sr_basic_names = set([str(name).lower() for name in sr_basic['subreddit_name']])
    not_drawing_srs = {name for name in sr_basic_names if name not in drawing_srs}
    # handling the sample size parameter
    if type(sample_size) is str:
        ratio = sample_size.split(':')
        sample_amount = int(float(ratio[0])*1.0 / float(ratio[1])*1.0 * len(drawing_srs))
    elif type(sample_size) is int and sample_size > 1:
        sample_amount = sample_size
    elif type(sample_size) is float and 0 < sample_size < 1:
        sample_amount = int(sample_size * len(not_drawing_srs))
    else:
        print("Current parameter type is not supported yet")
        return 1

    # now sampling the 'not_drawing_srs' population
    if balanced_sampling_based_sr_size:
        drawing_srs_df = sr_basic[sr_basic['subreddit_name'].str.lower().isin(drawing_srs)][
            ['subreddit_name', 'number_of_subscribers']]
        not_drawing_srs_df = sr_basic[sr_basic['subreddit_name'].str.lower().isin(not_drawing_srs)][
            ['subreddit_name', 'number_of_subscribers']]
        drawing_srs_dict = drawing_srs_df.set_index('subreddit_name').to_dict()['number_of_subscribers']
        not_drawing_srs_dict = not_drawing_srs_df.set_index('subreddit_name').to_dict()['number_of_subscribers']
        drawing_srs_dict = {k: int(v) for k, v in drawing_srs_dict.items()}
        not_drawing_srs_dict = {k: int(v) for k, v in not_drawing_srs_dict.items()}
        chosen_srs, diffs = _sample_srs_based_size(drawing_srs_data=drawing_srs_dict,
                                                    not_drawing_srs_data=not_drawing_srs_dict, size=sample_amount)
    else:
        np.random.seed(seed)
        sampling_idx = set(np.random.choice(a=range(0, len(not_drawing_srs)-1), size=sample_amount, replace=False))
        chosen_srs = [name for idx, name in enumerate(not_drawing_srs) if idx in sampling_idx]
    chosen_srs_full_info = sr_basic[sr_basic['subreddit_name'].str.lower().isin(chosen_srs)][['subreddit_name',
                                                                                              'number_of_subscribers',
                                                                                              'created_utc_as_date']]
    # returning results in a list format - each item is a tuple of 3. First is the name in lower-case letters,
    # second is the # of users in this SR and third is the creation date
    results = [(str(x[1]).lower(), x[2], x[3], 'not_drawing') for x in chosen_srs_full_info.itertuples()]
    # adding the drawing teams to the party and sending results
    chosen_srs_full_info2 = place_related_srs[place_related_srs['SR'].isin(drawing_srs)][['SR', 'num_of_users', 'creation_utc']]
    results2 = [(str(x[1]).lower(), x[2], x[3], 'drawing') for x in chosen_srs_full_info2.itertuples()]
    return results2 + results


def sr_sample_based_submissions(data_path, sample_size, threshold_to_define_as_drawing,
                                start_period='2017-01', end_period='2017-03', seed=1984):
    """
    sampling group of SRs to be later used for modeling. This group is in most cases the 'not-drawing' teams. We will
    use meta-data for sampling purposes, either internal meta-data or external one

    :param data_path: str
        location of all data needed for the algorithm
    :param sample_size: int or string of float
        how many SRs should be sampled. Can be represents as simple in (then, it should be in
        most cases >= len(not_drawing_SRs)). Can be float, and then it represents the % of SRs to choose. Can be string
        in the form of 'a:b' and then it represnts ratio between drawing and not-drawing team (e.g '2:1' means we will
        have not-drawing teams X2 compared to the drawing teams. '1:1' means pure balance situation)
    :param threshold_to_define_as_drawing: float
        a value to be used to define a SR as drawing, according to the value in the excel file. The value in the excel
        file represents our probability that a certain SR was drawing
    :param start_period: str
        the starting month of the data (format: YYYY-MM)
    :param end_period: str
        the ending month of the data (format: YYYY-MM)
    :param seed: int
        random seed to be used for the sampling (currently not being used)
    :return: list of tuples
        list where each item is a SRs and contains some information about it (represented in a tuple):
        [0] contains its name, [1] contains num_of_users, [2] contains 'creation_utc', [3] contains string ('drawing'
        or 'not_drawing')
    """
    start_time = datetime.datetime.now()

    # location of the csv
    csv_path = os.path.join(data_path, 'place_classifier_csvs')
    # finding all the relevant zip files in the 'data_path' directory
    submission_files = [f for f in os.listdir(csv_path) if re.match(r'RS.*\.csv', f)]
    # taking only the submissions files from 10-2016 to 03-2017 by default
    submission_files = [i for i in submission_files if 'RS_' + start_period + '.csv' <= i <= 'RS_' + end_period + '.csv']
    submission_files = sorted(submission_files)
    tot_srs_counter = collections.Counter()
    # iterating over each submission file
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=os.path.join(csv_path, cur_submission_file), encoding='utf-8')
        # making sure the the date is not after r/place started
        cur_submission_df = cur_submission_df[cur_submission_df['created_utc_as_date'] < '2017-03-29 00:00:00']
        cur_sr_submissions = cur_submission_df["subreddit"].str.lower()
        cur_srs_counter = collections.Counter(cur_sr_submissions)
        tot_srs_counter = tot_srs_counter + cur_srs_counter
    srs_counter_ordered = collections.OrderedDict(sorted(tot_srs_counter.items(), key=lambda t: t[1], reverse=True))
    # in order to see the first (and highest SR) - list(srs_counter_ordered.items())[0]
    # pulling out drawing srs information
    place_related_srs = pd.read_excel(io=os.path.join(data_path, 'subreddits_revealed',
                                                      'all_subredditts_based_atlas_and_submissions.xlsx'),
                                      sheet_name='Detailed list')
    drawing_srs = place_related_srs[(place_related_srs['trying_to_draw'] == 'Yes') |
                                    (place_related_srs['models_prediction'] > threshold_to_define_as_drawing)]['SR']
    # creating sets for the drawing/not-drawing teams
    drawing_srs = set([str(name).lower() for name in drawing_srs
                       if str(name).lower() in srs_counter_ordered.keys()])
    sr_basic_names = set([str(name).lower() for name in srs_counter_ordered.keys()])
    not_drawing_srs = {name for name in sr_basic_names if name not in drawing_srs}
    # handling the sample size parameter
    if type(sample_size) is str:
        ratio = sample_size.split(':')
        sample_amount = int(float(ratio[0])*1.0 / float(ratio[1])*1.0 * len(drawing_srs))
    elif type(sample_size) is int and sample_size > 1:
        sample_amount = sample_size
    elif type(sample_size) is float and 0 < sample_size < 1:
        sample_amount = int(sample_size * len(not_drawing_srs))
    else:
        print("Current parameter type is not supported yet")
        return 1
    # now sampling the 'not_drawing_srs' population

    drawing_srs_dict = {str(k).lower(): int(v) for k, v in srs_counter_ordered.items() if k in drawing_srs}
    not_drawing_srs_dict = {str(k).lower(): int(v) for k, v in srs_counter_ordered.items() if k in not_drawing_srs}
    chosen_srs, diffs = _sample_srs_based_size(drawing_srs_data=drawing_srs_dict,
                                               not_drawing_srs_data=not_drawing_srs_dict, size=sample_amount)

    # loading file with meta data about SRs, in order to return additional information about the returned SRs
    # this will include the SR size (users) and date of creation
    srs_meta_data = defaultdict(list)
    with open(os.path.join(data_path, 'srs_meta_data_102016_to_032017.json')) as f:
        for idx, line in enumerate(f):
            cur_line = json.loads(line)
            cur_sr_name = str(cur_line['display_name']).lower()
            # case we already 'met' this sr, we'll take the max out of all subscribers amount we see
            if cur_sr_name in srs_meta_data:
                srs_meta_data[cur_sr_name] = [max(cur_line['subscribers'], srs_meta_data[cur_sr_name][0]),
                                              cur_line['created']]

            elif cur_line['subscribers'] is not None:
                srs_meta_data[cur_sr_name] = [cur_line['subscribers'], cur_line['created']]
    sr_basic = pd.DataFrame.from_dict(data=srs_meta_data, orient='index')
    sr_basic.reset_index(inplace=True)
    sr_basic.columns = ['subreddit_name', 'number_of_subscribers', 'creation_epoch']
    sr_basic['number_of_subscribers'] = pd.to_numeric(sr_basic['number_of_subscribers'])

    # adding explicit timestamp to the data and filtering SRs which were created before r/place started
    sr_basic = sr_basic.assign(created_utc_as_date=pd.to_datetime(sr_basic['creation_epoch'], unit='s'))
    sr_basic = sr_basic[sr_basic['created_utc_as_date'] < '2017-03-31 00:00:00']

    # using the data we pulled out, we will connect between the two and return the needed information
    chosen_srs_full_info = sr_basic[sr_basic['subreddit_name'].str.lower().isin(chosen_srs)][['subreddit_name',
                                                                                              'number_of_subscribers',
                                                                                              'created_utc_as_date']]
    # returning results in a list format - each item is a tuple of 3. First is the name in lower-case letters,
    # second is the # of users in this SR and third is the creation date
    results = [(str(x[1]).lower(), x[2], x[3], 'not_drawing') for x in chosen_srs_full_info.itertuples()]
    # adding the drawing teams to the party and sending results
    chosen_srs_full_info2 = place_related_srs[place_related_srs['SR'].isin(drawing_srs)][['SR', 'num_of_users', 'creation_utc']]
    results2 = [(str(x[1]).lower(), x[2], x[3], 'drawing') for x in chosen_srs_full_info2.itertuples()]
    duration = (datetime.datetime.now() - start_time).seconds
    print("'sr_sample_based_submissions' function has ended, total of {} not drawing teams and {} of drawing teams were"
          " suggested to be used. Took us {} seconds".format(len(results), len(results2), duration))
    return results2 + results


def _sample_srs_based_size(drawing_srs_data, not_drawing_srs_data, size):
    """
    Sample observations in a smart way. The sampling tries to return the closets distribution possible to the original
    data got. This is a support function to the meta sampling functions (e.g., sr_sample_based_submissions)
    :param drawing_srs_data: list
        list of values related to the distribution values of the drawing team (i.e., list of the gorup size values
        of all the drawing teams in r/place)
    :param not_drawing_srs_data:  list
        similar list as the drawing_srs_data one, but here it is related to the not_drawing teams. Expected to be a
        much larger list than the drawing_srs_data one
    :param size: int
        how many instances to sample out of the not_drawing_srs_data group
    :return: set (of integers)
        set of int values, related to the indices of the not_drawing_srs_data (representing the chosen indices)

    Examples:
    >>> drawing_data = {'a': 5,'b': 10, 'c': 8, 'd': 3, 'e': 22}
    >>> not_drawing_data = {'z': 40, 'x': 13, 'w': 22, 'v': 45, 'q': 8, 'r': 9, 's': 16, 't': 45, 'm': 98, 'n': 104}
    >>> _sample_srs_based_size(drawing_srs_data=drawing_data, not_drawing_srs_data=not_drawing_data, size=len(drawing_data))
    """
    if size > len(not_drawing_srs_data):
        print("Not possible to sample a subset larger than the origianl group size. 'size' must be smaller than the"
              "size of not_drawing_srs_data gropu. Try again please")
        return -1
    # first value is the original index, second one is the value. It will be ordered lowest to highest
    #drawing_srs_mapping = [(idx, value) for idx, value in enumerate(drawing_srs_data)]
    #not_drawing_srs_mapping = [(idx, value) for idx, value in enumerate(not_drawing_srs_data)]
    drawing_srs_mapping = list(OrderedDict(sorted(drawing_srs_data.items(), key=lambda x: x[1], reverse=False)).items())
    not_drawing_srs_mapping = list(OrderedDict(sorted(not_drawing_srs_data.items(), key=lambda x: x[1], reverse=False)).items())
    # some variables to handle the loop and the lookup process
    drawing_cur_idx = 0
    not_drawing_cur_idx = 0
    diffs = []
    chosen_srs = []
    # looping over and over till the list of indices we return is big enough
    while len(chosen_srs) < size:
        drawing_cur_value = drawing_srs_mapping[drawing_cur_idx][1]
        not_drawing_cur_value = not_drawing_srs_mapping[not_drawing_cur_idx][1]
        highest_value_limit_flag = False
        # inner loop, which looks for the best fit for the current candidate value
        while not_drawing_cur_value < drawing_cur_value and not highest_value_limit_flag:
            not_drawing_cur_idx += 1
            try:
                not_drawing_cur_value = not_drawing_srs_mapping[not_drawing_cur_idx][1]
            # case the highest value in the not_drawing_list is lower than the value we look for
            except IndexError:
                not_drawing_cur_idx -= 1
                not_drawing_cur_value = not_drawing_srs_mapping[not_drawing_cur_idx][1]
                highest_value_limit_flag = True
                print("Along the _sample_srs_based_size function, highest_value_limit_flag turned to True once")
        # deciding which side to take (higher/lower than the the comparison to
        if not_drawing_cur_idx > 0:
            lower_option = not_drawing_srs_mapping[not_drawing_cur_idx-1][1]
            upper_option = not_drawing_cur_value
            chosen_idx = (not_drawing_cur_idx - 1) if abs(drawing_cur_value-lower_option) <= \
                                                      abs(drawing_cur_value-upper_option) else not_drawing_cur_idx
        else:
            chosen_idx = 0
        # adding the relevant name to the list of chosen indices
        chosen_srs.append(not_drawing_srs_mapping[chosen_idx][0])
        diffs.append(abs(drawing_cur_value - not_drawing_srs_mapping[chosen_idx][1]))
        # taking the instance we chose of of the big list, since we do not want to have duplications
        not_drawing_srs_mapping.pop(chosen_idx)
        # since we took out an instance, we need to update the index of the not_drawing group
        not_drawing_cur_idx = max(0, not_drawing_cur_idx-1)
        # case we haven't reached the end of the drawing group, we will continue iterating over it
        if drawing_cur_idx < len(drawing_srs_data) - 1:
            drawing_cur_idx += 1
        # case we did reach the end of this group, we will start a new "cycle" of sampling, but not from the begining,
        # since then our sample would be biased towrds small submissions amount SRs
        else:
            drawing_cur_idx = int(len(drawing_srs_mapping) * 1.0 / 2.0)
            not_drawing_cur_idx = 0
    print("Summary of the _sample_srs_based_size function: we sampled {} SRs."
          "Total difference according to the data is: {}".format(len(chosen_srs), sum(diffs)))
    return [sr.lower() for sr in chosen_srs], diffs


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    general_loader(data_path=data_path, saving_path='/data/work/data/reddit_place/' + 'place_classifier_csvs',
                   load_only_columns_subset=True)
    #res = sr_sample_based_subscribers(data_path=data_path, sample_size='1:1', threshold_to_define_as_drawing=0.7)
    #res = sr_sample_based_submissions(data_path=data_path, sample_size='1.1:1', start_peoriod='2017-01',
    #                                  end_period='2017-03', threshold_to_define_as_drawing=0.7)
    duration = (datetime.datetime.now() - start_time).seconds
    #print("Finished. Res size is:{}, took us: {}". format(len(res), duration))
    print("Finished. Took us: {} seconds".format(duration))



