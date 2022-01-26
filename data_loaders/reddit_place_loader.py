# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 6.11.2018

import pandas as pd
import re
import sys
import numpy as np

###################################################### Configurations ##################################################
data_path = '/home/isabrah/reddit_data/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\sr_relations\\'
use_comments_data = True
test_data_logic = None   # either 'atlas_based' or 'submission_based' or None
########################################################################################################################

if __name__ == "__main__":
    # reading the data again from the data location source
    submission_data = pd.read_json(data_path + 'RS_31-3_to_4-4_2017.txt', lines=True)
    comments_data = pd.read_json(data_path + 'RC_31-3_to_4-4_2017.txt', lines=True)

    # loading the subreddits data (now it is a df, we'll take a single column)
    subreddits_df = pd.read_csv(data_path + 'all_subreddits_revealed.csv')
    subreddits_list = list(subreddits_df['sr'])
    # adding explicit timestamp to the data
    submission_data = submission_data.assign(created_utc_as_date=pd.to_datetime(submission_data['created_utc'],
                                                                                unit='s'))
    comments_data = comments_data.assign(created_utc_as_date=pd.to_datetime(comments_data['created_utc'], unit='s'))
    # sort the subreddit data by time
    submission_data.sort_values(by='created_utc_as_date', ascending=True, inplace=True)
    comments_data.sort_values(by='created_utc_as_date', ascending=True, inplace=True)

    # filter the data again, now taking only few specific intresting columns from each dataset
    submissions_interesting_col = ["created_utc_as_date", "author", "subreddit", "title", "selftext", "num_comments",
                                   "view_count", "permalink", "score", "id"]
    submissions_shrinked = submission_data[submissions_interesting_col]
    comments_interesting_col = ["created_utc_as_date", "author", "subreddit", "body", "score", "id",
                                "link_id", "parent_id"]
    comments_shrinked = comments_data[comments_interesting_col]

    # creating a subset which is only comments and submissions taken from the list of relevant subreddits (~2300 srs)
    submissions_subset = submissions_shrinked[submissions_shrinked['subreddit'].str.lower().isin(subreddits_list)]
    comments_subset = comments_shrinked[comments_shrinked['subreddit'].str.lower().isin(subreddits_list)]
    print("Current submission data dimension is {}. "
          "Current comments data dimension is {}".format(submissions_shrinked.shape, comments_subset.shape))

    # now doing 2 cycles on the data (submissions and comments) to filter out only rows directly related to r/place
    # First cycle - submissions data
    indicator_words = ['r/place', '@Place', 'canvas', 'pixel', 'flag', '/r/place']
    regex = '|'.join(indicator_words)
    relevant_submissions_idx = set()
    for index, row in submissions_subset.iterrows():
        cur_title = row['title']
        cur_selftext = row['selftext']
        title_res = re.search(regex, cur_title, re.IGNORECASE)
        selftext_res = re.search(regex, cur_selftext, re.IGNORECASE)
        if title_res is not None or selftext_res is not None:
            relevant_submissions_idx.add(index)
    relevant_submissions_ids = {"t3_"+id for id in submissions_shrinked.loc[relevant_submissions_idx]["id"]}

    # First cycle - comments data
    relevant_comments_idx = set()
    for index, row in comments_subset.iterrows():
        cur_body = row['body']
        cur_linkid = row['link_id']
        body_res = re.search(regex, cur_body, re.IGNORECASE)
        if body_res is not None or cur_linkid in relevant_submissions_ids:
            relevant_comments_idx.add(index)
    print("Currently we have {} rows in the submissions data and {} "
          "rows in the comments data".format(len(relevant_submissions_idx), len(relevant_comments_idx)))

    # Second cycle - submissions data. Adding items based on comments we found
    comments_temp_subset = comments_subset.loc[relevant_comments_idx]
    comments_temp_linkid = comments_temp_subset["link_id"]
    comments_temp_linkid_list = [l[3:] for l in comments_temp_linkid]
    submissions_temp_subset = submissions_subset[submissions_subset['id'].isin(comments_temp_linkid_list)]
    index_to_add = set(submissions_temp_subset.index.values)
    ids_to_add = set(['t3_'+i for i in submissions_temp_subset['id']])
    relevant_submissions_idx.update(index_to_add)
    relevant_submissions_ids.update(ids_to_add)

    comments_temp_subset = comments_subset[comments_subset['link_id'].isin(list(relevant_submissions_ids))]
    index_to_add = set(comments_temp_subset.index.values)
    relevant_comments_idx.update(index_to_add)

    print("Currently we have {} rows in the submissions data and {} "
          "rows in the comments data".format(len(relevant_submissions_idx), len(relevant_comments_idx)))

    # saving the objects as hdf files
    submissions_to_save = submissions_subset.loc[relevant_submissions_idx].copy()
    comments_to_save = comments_subset.loc[relevant_comments_idx].copy()
    submissions_to_save.to_hdf(path_or_buf='/home/isabrah/reddit_data/submission_shrinked.hdf',
                               key='submission_shrinked')
    comments_to_save.to_hdf(path_or_buf='/home/isabrah/reddit_data/comments_shrinked.hdf',
                            key='comments_shrinked')