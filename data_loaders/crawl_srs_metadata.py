# Authors: Avrahami Israeli (isabrah)
# Python version: 3.6
# Last update: 6.11.2018

import os
import re
import sys
import pandas as pd
import datetime
import praw
from prawcore.exceptions import Forbidden, NotFound
import json
import numpy as np

data_path = '/home/reddit_data/' if sys.platform == 'linux' \
    else 'C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\PhD\\reddit canvas\\data\\'


def crawl_srs_meta_data(reddit_obj):
    """
    crawling meta data regaring SRs in reddit. Crawling is based on all SRs found in the csv files (containing all the
    submissions along a period of time)
    :param reddit_obj: praw.Reddit
        the API object which will be used for crawling
    :return: None
        saving all results to files
    """
    start_time = datetime.datetime.now()
    csvs_location = data_path + 'place_classifier_csvs/' if sys.platform == 'linux' \
        else data_path + 'place_classifier_csvs\\'
    #submission_files = [f for f in os.listdir(csvs_location) if re.match(r'RS.*\.csv', f)]
    submission_files = ['RS_2016-10.csv', 'RS_2016-11.csv', 'RS_2016-12.csv',
                        'RS_2017-01.csv', 'RS_2017-02.csv', 'RS_2017-03.csv']
    print("{} files have been found and will be handled".format(len(submission_files)))
    srs_found = set()
    # looping over each submission file, pulling out the SR name of each submission
    for subm_idx, cur_submission_file in enumerate(submission_files):
        cur_submission_df = pd.read_csv(filepath_or_buffer=csvs_location + cur_submission_file)
        cur_srs_found = set(cur_submission_df["subreddit"].str.lower())
        srs_found = srs_found.union(cur_srs_found)
    duration = (datetime.datetime.now() - start_time).seconds
    print("Total of {} srs were found. Up to now, took us {} sec. Moving to crawling phase".format(len(srs_found), duration))

    # converting the SRs found to a list that will be ordered without the nan object
    srs_found = sorted([sr for sr in srs_found if sr is not np.nan])
    srs_metadata = []
    for idx, cur_sr_name in enumerate(srs_found):
        if type(cur_sr_name) is not str:
            continue
        try:
            cur_sr_obj = reddit_obj.subreddit(cur_sr_name)
        except Forbidden as e:
            continue
        try:
            cur_sr_obj.subscribers
        except (NotFound, Forbidden) as e:
            continue
        cur_sr_info = vars(cur_sr_obj)
        # removing a problematic key from the list (it is a Reddit object key)
        cur_sr_info.pop('_reddit', None)
        # adding the SR data to the list of data
        srs_metadata.append(cur_sr_info)
        #cur_sr_subscribers = cur_sr_info['subscribers']
        # each 5000 SRs, we will save results and print to screen the status
        if idx % 5000 == 0:
            saving_loc = data_path + '/srs_meta_data.json' if sys.platform == 'linux' else data_path + '\\srs_meta_data.json'
            with open(saving_loc, 'a') as f:
                for sr in srs_metadata:
                    json.dump(sr, f)
                    f.write('\n')
            duration = (datetime.datetime.now() - start_time).seconds
            print("We are along the crawling phase. Tried to crawl up to now {} SRs, found {}."
                  "took us up to now {} sec".format(idx, len(srs_metadata), duration))
            srs_metadata = []


if __name__ == "__main__":
    reddit = praw.Reddit(client_id='FnSwswueosOi_g',
                         client_secret='kcUc1ZMxUDLHgO4gzdFCdca79xk',
                         password='gefenp4',
                         user_agent='Avrahami_crawling_data',
                         username='avrahami_isr')
    crawl_srs_meta_data(reddit_obj=reddit)
