# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 6.11.2018

import datetime
import pandas as pd
import re
import sys
import os
import bz2
import json
import csv
import lzma

###################################################### Configurations ##################################################
data_path = '/home/isabrah/reddit_data/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'
data_to_process = 'comments'   # can be either 'submission' / 'comments' / 'both'
regex_required = 'fake[\s_-]*news'
included_years = [2016, 2015, 2014, 2013, 2012, 2011, 2010]
#included_years = [2017]
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


def regex_based_loader(data_path, sr_to_include=None, saving_path=os.getcwd(), load_only_columns_subset=False):
    """
    loading reddit data, based on the zipped files from here - https://files.pushshift.io/reddit/
    the process converts this files into csv, after filtering some columns (if load_only_columns_subset=True),
    adding a date column and filtering a regex given. Logic of this function is almost identical to the one in the
    'general_loader' function

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
    submission_files = [sf for sf in submission_files if any(str(year) in sf for year in included_years)]
    comments_files = [cf for cf in comments_files if any(str(year) in cf for year in included_years)]
    submission_files = sorted(submission_files)
    comments_files = sorted(comments_files)
    submissions_interesting_col = ["created_utc_as_date", "author", "subreddit", "title", "selftext", "num_comments",
                                   "permalink", "score", "id", "thumbnail"]
    comments_interesting_col = ["created_utc_as_date", "author", "subreddit", "body", "score", "id",
                                "link_id", "parent_id", "thumbnail"]
    meta_data_file = saving_path + '/' if sys.platform == 'linux' else saving_path + '\\'
    meta_data_file += 'meta_data_desired_regex_filtered.csv'
    # writing the header for the meta data file
    m = open(meta_data_file, 'a')
    with m as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=['file', 'tot_lines', 'relvent_lines'])
        dict_writer.writeheader()
    m.close()
    # looping over each file of the submission/comment and handling it
    if data_to_process == 'submission' or data_to_process == 'both':
        # for cur_submission_file in submission_files[0]:
        for subm_idx, cur_submission_file in enumerate(submission_files):
            # we expect to see inside the zip, a file called exactly the same as the zip one, besides the ending 'bz2'
            # or 'xz
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
                cur_title = cur_line['title']
                cur_selftext = cur_line['selftext']
                title_res = re.search(regex_required, cur_title, re.IGNORECASE)
                selftext_res = re.search(regex_required, cur_selftext, re.IGNORECASE)
                if title_res is None and selftext_res is None:
                    continue
                cur_line['created_utc_as_date'] = str(pd.to_datetime(int(cur_line['created_utc']), unit='s'))
                if load_only_columns_subset:
                    line_shrinked = dict((k, cur_line[k]) if k in cur_line else (k, None) for k in submissions_interesting_col)
                # we still define this 'line_shrinked' also in cases when we want to have all columns, since in some
                # cases there are redundant columns appear in the zip original files
                else:
                    line_shrinked = dict((k, cur_line[k]) if k in cur_line else (k, None) for k in submission_columns)
                submissions.append(line_shrinked)
            # saving the data to a file. Currently it is as a csv format (found it as the most useful one)
            full_file_name = saving_path + '/' if sys.platform == 'linux' else saving_path + '\\'
            full_file_name += cur_submission_file[0:7] + '_desired_regex_filtered.csv'
            f = open(full_file_name, 'a')
            # only in case we found some relevant post - we'll add it to the file
            if len(submissions) > 0:
                keys = submissions[0].keys()
                with f as output_file:
                    dict_writer = csv.DictWriter(output_file, keys)
                    # adding a header only if the file is a new one
                    if os.stat(full_file_name).st_size == 0:
                        dict_writer.writeheader()
                    dict_writer.writerows(submissions)
                f.close()
            m = open(meta_data_file, 'a')
            with m as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=['file', 'tot_lines', 'relvent_lines'])
                dict_writer.writerow({'file': cur_submission_file, 'tot_lines': inner_idx, 'relvent_lines': len(submissions)})
            m.close()
            duration = (datetime.datetime.now() - start_time).seconds
            print("Finished handling the {}'th submission file called {}. Took us up to now: {} seconds. "
                  "Current submission size is {}".format(subm_idx + 1, cur_submission_file, duration, len(submissions)))
    if data_to_process == 'comments' or data_to_process == 'both':
        for comm_idx, cur_comments_file in enumerate(comments_files):
            if cur_comments_file.endswith('bz2'):
                zipped_comment = bz2.BZ2File(comments_files_path + cur_comments_file, 'r')
            else:
                zipped_comment = lzma.open(comments_files_path + cur_comments_file, mode='r')
            # looping over each row in the comments data
            comments = []
            for inner_idx, line in enumerate(zipped_comment):
                try:
                    cur_line = json.loads(line.decode('UTF-8'))
                except json.decoder.JSONDecodeError:
                    continue
                cur_body = cur_line['body']
                body_res = re.search(regex_required, cur_body, re.IGNORECASE)
                if body_res is None:
                    continue
                cur_line['created_utc_as_date'] = pd.to_datetime(int(cur_line['created_utc']), unit='s')
                if load_only_columns_subset:
                    line_shrinked = dict((k, cur_line[k]) if k in cur_line else (k, None) for k in comments_interesting_col)
                # we still define this 'line_shrinked' also in cases when we want to have all columns, since in some
                # cases there are redundant columns appear in the zip original files
                else:
                    line_shrinked = dict((k, cur_line[k]) if k in cur_line else (k, None) for k in comments_columns)
                comments.append(line_shrinked)
            # saving the data to a file. Currently it is as a csv format (found it as the most useful one)
            full_file_name = saving_path + '/' if sys.platform == 'linux' else saving_path + '\\'
            full_file_name += cur_comments_file[0:7] + '_desired_regex_filtered.csv'
            f = open(full_file_name, 'a')
            if len(comments) > 0:
                keys = comments[0].keys()
                with f as output_file:
                    dict_writer = csv.DictWriter(output_file, keys)
                    # adding a header only if the file is a new one
                    if os.stat(full_file_name).st_size == 0:
                        dict_writer.writeheader()
                    dict_writer.writerows(comments)
                f.close()
            m = open(meta_data_file, 'a')
            with m as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=['file', 'tot_lines', 'relvent_lines'])
                dict_writer.writerow({'file': cur_comments_file,
                                      'tot_lines': inner_idx, 'relvent_lines': len(comments)})
            m.close()
            duration = (datetime.datetime.now() - start_time).seconds
            print("Finished handling the {}'th comments file. Took us up to now: {} seconds. "
                  "Current comments size is {}".format(comm_idx+1, duration, len(comments)))


if __name__ == "__main__":
    regex_based_loader(data_path=data_path)

####################### EXTRA CODE ##############
'''
# this is in case we want to pull out some statistics regararind each file (distinct users, distinct SR in each file)
import os
import pandas as pd
os.chdir('/home/isabrah/reddit_pycharm_proj_with_own_pc/data_loaders/fake_news_data')
cur_df = pd.read_csv('RC_2018_desired_regex_filtered.csv')
cur_df['month'] = pd.DatetimeIndex(cur_df['created_utc_as_date']).month
unique_srs = cur_df.groupby(['month'])['subreddit'].nunique()
unique_users = cur_df.groupby(['month'])['author'].nunique()
result = pd.concat([unique_srs, unique_users], axis=1)
print(result)
'''