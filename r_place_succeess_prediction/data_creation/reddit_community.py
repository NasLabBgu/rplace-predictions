import sys
import gc
import pickle
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import copy
import warnings
import math
import random
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.data_creation.data_creation_utils import get_indices_inside_artwork, extract_liwc_occurrences
from r_place_success_analysis.data_creation.community_social_network import CommunitySocialNetwork
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
from r_place_drawing_classifier.utils import get_submissions_subset, get_comments_subset

exper_submissions_raw_file = "/data/work/data/reddit_place/place_classifier_csvs/RS_r_place_duration.p"
exper_comments_raw_file = "/data/work/data/reddit_place/place_classifier_csvs/RC_r_place_duration.p"


class RedditCommunity(object):
    def __init__(self, name, creation_utc, survived, num_users, meta_features_pre_exper,
                 struct_features_pre_exper, pixels_path):
        self.name = name
        # this is a list of dict, holding all information about the area of the artwork
        self.pixels_path = pixels_path
        self.creation_utc = creation_utc
        self.survived = survived
        self.num_users = num_users
        self.meta_features = meta_features_pre_exper,
        self.struct_features = struct_features_pre_exper
        # the next two variables are calculated by the ??? and the ??? functions
        self.exper_meta_features = defaultdict()
        self.exper_struct_features = defaultdict()

        # the next two variables are assigned by the 'extract_seed_users' and the 'extract_potential_users' function
        self.seed_users = None
        self.potential_users = None
        # the next 4 variables are assigned by the 'extract_while_exper_textual_data' function
        self.exper_submissions_as_list = None
        self.exper_submissions_as_tokens = None
        self.exper_comments_as_list = None
        self.exper_comments_as_tokens = None
        self.exper_submissions_tokens_dict = defaultdict(int)
        self.exper_comments_tokens_dict = defaultdict(int)

        # the next 6 variables are assigned by the 'extract_pre_exper_textual_data' function
        self.submissions_as_list = None
        self.submissions_as_tokens = None
        self.comments_as_list = None
        self.comments_as_tokens = None
        self.submissions_tokens_dict = None
        self.comments_tokens_dict = None

        self.liwc_features = None
        self.exper_liwc_features = None

        # the next 2 variables are assigned by the 'prepare_text_for_bert' function
        self.text_data_for_bert = None
        self.exper_text_data_for_bert = None

    def extract_seed_users(self, tiles_placement, user_subreddits_dict=None, consider_only_writting_users=False):
        seed_users = set()
        # looping over each artwork
        for cur_pixels_path in self.pixels_path:
            tiles_placement_cur_artwork = \
                self.get_tiles_placement_inside_paths(tiles_placement=tiles_placement,
                                                      pixels_path=[cur_pixels_path], min_percentage_in_path=0.5)
            tiles_placement_cur_artwork_max_ts = self.get_max_ts_tiles_placement(tiles_placement_cur_artwork)
            seed_users.update(set(tiles_placement_cur_artwork_max_ts['original_username']))
        # saving the seed users and returning the seed users anyway
        self.seed_users = seed_users
        # in case we want to take only users that wrote something in the community (in the past 3 months)
        if consider_only_writting_users:
            new_seed_users = list()
            for cur_u in self.seed_users:
                if cur_u in user_subreddits_dict and self.name in user_subreddits_dict[cur_u]:
                    new_seed_users.append(cur_u)
            self.seed_users = set(new_seed_users)
        return self.seed_users

    def extract_potential_users(self, tiles_placement, user_subreddits_dict, min_ts=1491151733000):
        # cur min_ts is 24 hours back from the end of r/place
        # we first pull out the potential users that wrote something in the given SR (even one comment)
        potential_users = list()
        for cur_u_name, cur_u_srs in user_subreddits_dict.items():
            if self.name in cur_u_srs:
                potential_users.append(cur_u_name)
        potential_users = set(potential_users)
        # now for every user, we will check if he/she added a pixel in the area of the community we deal with
        # we filter the tiles_placement df so things will be quicker. The timestamp filter is inside the function
        potential_users_filtered = dict()
        tiles_placement_filtered = copy.deepcopy(tiles_placement)
        tiles_placement_filtered = \
            tiles_placement_filtered[tiles_placement_filtered['original_username'].isin(potential_users)]
        tiles_placement_filtered = self.get_tiles_placement_inside_paths(tiles_placement=tiles_placement_filtered,
                                                                         min_percentage_in_path=0.5)
        for cur_p_user in potential_users:
            cur_p_user_pixels_cnt = \
                self.pixels_cnt_per_user(tiles_placement=tiles_placement_filtered, user_name=cur_p_user, min_ts=min_ts)
            if cur_p_user_pixels_cnt > 0:
                potential_users_filtered[cur_p_user] = cur_p_user_pixels_cnt
        # sorting the list according to the amount of pixels
        potential_users_filtered = {k: potential_users_filtered[k] for k in sorted(potential_users_filtered,
                                                                                   key=potential_users_filtered.get,
                                                                                   reverse=True)}
        self.potential_users = potential_users_filtered
        return potential_users_filtered

    def extract_pre_exper_textual_data(self, pre_exper_sr_obj):
        self.submissions_as_list = pre_exper_sr_obj.submissions_as_list
        self.comments_as_list = pre_exper_sr_obj.comments_as_list
        self.submissions_as_tokens = pre_exper_sr_obj.submissions_as_tokens
        self.comments_as_tokens = pre_exper_sr_obj.comments_as_tokens
        self.submissions_tokens_dict = pre_exper_sr_obj.submissions_tokens_dict
        self.comments_tokens_dict = pre_exper_sr_obj.comments_tokens_dict
        return 0

    def extract_while_exper_textual_data(self, submission_dp_obj, comments_dp_obj):
        """
        Most of the ideas and concepts are taken from the main_srs_creation.py file
        (under the r_place_drawing_classifier)

        :param submission_dp_obj:
        :param comments_dp_obj:
        :return: None

        cur_sr_submission = get_submissions_subset(files_path="/data/work/data/reddit_place/place_classifier_csvs",
                                                   srs_to_include=[self.name], start_month="2017-03",
                                                   end_month="2017-04",
                                                   min_utc='2017-03-29 00:00:00', max_utc='2017-04-04 00:00:00')
        cur_sr_comments = get_comments_subset(files_path="/data/work/data/reddit_place/place_classifier_csvs",
                                              srs_to_include=[self.name], start_month="2017-03", end_month="2017-04",
                                              min_utc='2017-03-29 00:00:00', max_utc='2017-04-04 00:00:00')
        """
        all_submissions = pickle.load(open(exper_submissions_raw_file, "rb"))
        all_comments = pickle.load(open(exper_comments_raw_file, "rb"))
        cur_sr_submission = all_submissions[all_submissions["subreddit"].str.lower() == self.name]
        cur_sr_comments = all_comments[all_comments["subreddit"].str.lower() == self.name]
        # case there are no relevant submissions to this sr
        if cur_sr_submission.shape[0] == 0:
            self.exper_submissions_as_list = []
            self.exper_comments_as_list = []
            self.exper_submissions_as_tokens = []
            self.exper_comments_as_tokens = []
            return
        # pulling out the meta data about the SR (the iloc[0] is used only to convert it into Series)
        cur_sr_submission_after_dp, submission_text = submission_dp_obj.data_pre_process(reddit_df=cur_sr_submission)
        # updating the object with the list of submissions
        self.exper_submissions_as_list = submission_text
        del cur_sr_submission
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
        self.exper_submissions_as_tokens = full_tok_text
        del full_tok_text

        submission_ids = set(['t3_' + sub_id for sub_id in cur_sr_submission_after_dp['id']])
        # second, we filter the comments, so only ones which are relevant to the submissions dataset will appear
        # (this is due to the fact that we have already filtered a lot of submissions in pre step)
        cur_sr_comments = cur_sr_comments[cur_sr_comments['link_id'].isin(submission_ids)]
        cur_sr_comments_after_dp, comments_text = comments_dp_obj.data_pre_process(reddit_df=cur_sr_comments)
        del cur_sr_comments
        self.exper_comments_as_list = comments_text
        full_tok_text = []
        for s in comments_text:
            if type(s[1]) is str:
                sample_for_tokenizer = comments_dp_obj.mark_urls(s[1], marking_method='tag')[0]
                cur_tok_words = comments_dp_obj.tokenize_text(sample=sample_for_tokenizer, convert_to_lemmas=False,
                                                              break_to_sents=True)
                full_tok_text.append(cur_tok_words)
        self.exper_comments_as_tokens = full_tok_text
        del full_tok_text
        gc.collect()
        # creating the comments/submissions aggregated dicts (# of occurrences per word)
        self.update_exper_words_dicts(update_only_submissions_dict=True)
        return

    def extract_exper_meta_features(self):
        """
        Create meta features, by extracting them from the raw data.
        Most of the code is taken from the meta features creation process of the pre-experiment data (code is under:
        'reddit-tools/sr_classifier/sub_reddit.py"


        :return:
        """
        all_submissions = pickle.load(open(exper_submissions_raw_file, "rb"))
        all_comments = pickle.load(open(exper_comments_raw_file, "rb"))
        submission_data = all_submissions[all_submissions["subreddit"].str.lower() == self.name]
        comments_data = all_comments[all_comments["subreddit"].str.lower() == self.name]

        self.exper_meta_features['submission_amount'] = submission_data.shape[0]
        # starting calculating all relevant features to the submissions ones
        titles_len = []
        selftexts_len = []
        scores_values = []
        empty_selftext_cnt = 0
        deleted_or_removed_amount = 0
        deleted_authors = 0
        submissions_writers = defaultdict(int)
        for cur_idx, cur_sub_row in submission_data.iterrows():
            try:
                titles_len.append(len(cur_sub_row['title']))
            except TypeError:
                pass
            if type(cur_sub_row['score']) is int or type(cur_sub_row['score']) is float:
                scores_values.append(cur_sub_row['score'])
            if type(cur_sub_row['selftext']) is str and \
                    cur_sub_row['selftext'] != '[deleted]' and cur_sub_row['selftext'] != '[removed]':
                selftexts_len.append(len(cur_sub_row['selftext']))
            if type(cur_sub_row['author']) is str and cur_sub_row['author'] != '[deleted]':
                submissions_writers[cur_sub_row['author']] += 1
            if type(cur_sub_row['selftext']) is float and np.isnan(cur_sub_row['selftext']):
                empty_selftext_cnt += 1
            if cur_sub_row['selftext'] in {'[deleted]', '[removed]'}:
                deleted_or_removed_amount += 1
            if type(cur_sub_row['author']) is str:
                if cur_sub_row['author'] != '[deleted]':
                    submissions_writers[cur_sub_row['author']] += 1
                else:
                    deleted_authors += 1
        # now, saving results to the sr object
        self.exper_meta_features['avg_submission_title_length'] = np.mean(titles_len) if len(titles_len) > 0 else 0
        self.exper_meta_features['median_submission_title_length'] = \
            np.median(titles_len) if len(titles_len) > 0 else 0
        self.exper_meta_features['avg_submission_selftext_length'] = \
            np.mean(selftexts_len) if len(selftexts_len) > 0 else 0
        self.exper_meta_features['median_submission_selftext_length'] = \
            np.median(selftexts_len) if len(selftexts_len) > 0 else 0
        self.exper_meta_features['submission_average_score'] = np.mean(scores_values) if len(
            scores_values) > 0 else 0
        self.exper_meta_features['submission_median_score'] = np.median(scores_values) if len(
            scores_values) > 0 else 0
        # % of empty selftext exists in submissions + # % of deleted or removed submissions
        try:
            self.exper_meta_features['empty_selftext_ratio'] = empty_selftext_cnt * 1.0 / submission_data.shape[0]
            self.exper_meta_features['deleted_removed_submission_ratio'] = \
                deleted_or_removed_amount * 1.0 / submission_data.shape[0]
        except ZeroDivisionError:
            self.exper_meta_features['empty_selftext_ratio'] = 0.0
            self.exper_meta_features['deleted_removed_submission_ratio'] = 0.0
        self.exper_meta_features['submission_distinct_users'] = len(submissions_writers.keys())
        self.exper_meta_features['submission_users_std'] = \
            np.std(list(submissions_writers.values())) if len(submissions_writers.values()) > 0 else None
        self.exper_meta_features['average_submission_per_user'] = \
            (submission_data.shape[0] + 1) * 1.0 / (self.exper_meta_features['submission_distinct_users'] + 1)
        self.exper_meta_features['median_submission_per_user'] = \
            np.median(list(submissions_writers.values())) if len(submissions_writers) > 0 else 0
        self.exper_meta_features['submission_amount_normalized'] = \
            submission_data.shape[0] * 1.0 / len(submissions_writers) if len(submissions_writers) > 0 else 0
        # comments related features - only relevant if we have comments data
        if comments_data is not None and comments_data.shape[0] > 0:
            body_len = []
            scores_values_comments = []
            deleted_or_removed_amount_comments = 0
            comments_writers = defaultdict(int)
            commented_comments = defaultdict(int)
            commented_submissions = defaultdict(int)
            commenting_a_submission_cnt = 0
            self.exper_meta_features['comments_amount'] = comments_data.shape[0]
            for cur_idx, cur_comm_row in comments_data.iterrows():
                try:
                    body_len.append(len(cur_comm_row['body']))
                except TypeError:
                    pass
                if type(cur_comm_row['score']) is int or type(cur_comm_row['score']) is float:
                    scores_values_comments.append(cur_comm_row['score'])
                if type(cur_comm_row['author']) is str:
                    if cur_comm_row['author'] != '[deleted]':
                        comments_writers[cur_comm_row['author']] += 1
                    else:
                        deleted_authors += 1
                try:
                    if cur_comm_row['parent_id'].startswith('t3'):
                        commenting_a_submission_cnt += 1
                    elif cur_comm_row['parent_id'].startswith('t1'):
                        commented_comments[cur_comm_row['parent_id']] += 1
                except AttributeError:
                    pass
                if type(cur_comm_row['link_id']) is str:
                    commented_submissions[cur_comm_row['link_id']] += 1
                if cur_comm_row['body'] in {'[deleted]', '[removed]'}:
                    deleted_or_removed_amount_comments += 1
            # general comments features
            self.exper_meta_features['avg_comments_length'] = np.mean(body_len) if len(body_len) > 0 else 0
            self.exper_meta_features['median_comments_length'] = np.median(body_len) if len(body_len) > 0 else 0
            self.exper_meta_features['comments_average_score'] = \
                np.average(scores_values_comments) if len(scores_values_comments) > 0 else 0
            self.exper_meta_features['comments_median_score'] = \
                np.median(scores_values_comments) if len(scores_values_comments) > 0 else 0
            self.exper_meta_features['median_comments_per_user'] = \
                np.median(list(comments_writers.values())) if len(submissions_writers) > 0 else 0
            self.exper_meta_features['deleted_removed_comments_ratio'] = deleted_or_removed_amount_comments * 1.0 / \
                                                                         comments_data.shape[0] if len(
                comments_writers) > 0 else 0
            self.exper_meta_features['comments_users_std'] = \
                np.std(list(comments_writers.values())) if len(comments_writers.values()) > 0 else None
            self.exper_meta_features['comments_amount_normalized'] = \
                comments_data.shape[0] * 1.0 / len(comments_writers) if len(comments_writers) > 0 else 0
            # comments features which are related to the submissions as well / users
            # adding 1 to both denominator and numerator to overcome division by zero errors
            self.exper_meta_features['comments_submission_ratio'] = (comments_data.shape[0] + 1) * 1.0 / \
                                                                     (submission_data.shape[0] + 1)
            self.exper_meta_features['submission_to_comments_users_ratio'] = \
                (len(comments_writers.keys()) + 1) * 1.0 / \
                (self.exper_meta_features['submission_distinct_users'] + 1) * 1.0
            self.exper_meta_features['distinct_comments_to_submission_ratio'] = \
                (len(commented_submissions.keys()) + 1) * 1.0 / (submission_data.shape[0] + 1) * 1.0
            self.exper_meta_features['distinct_comments_to_comments_ratio'] = \
                (len(commented_comments.keys()) + 1) * 1.0 / (comments_data.shape[0] + 1) * 1.0
            self.exper_meta_features['submission_to_comments_words_used_ratio'] = \
                (self.exper_meta_features['avg_submission_selftext_length'] + 1) * 1.0 / \
                (self.exper_meta_features['avg_comments_length'] + 1)
            self.exper_meta_features['users_amount'] = \
                len(set(comments_writers.keys()).union(set(submissions_writers.keys())))
            if self.exper_meta_features['users_amount'] > 0:
                self.exper_meta_features['deleted_users_normalized'] = \
                    deleted_authors * 1.0 / self.exper_meta_features['users_amount']
            else:
                self.exper_meta_features['users_amount'] = 0
        # anyway, adding the users_amount feature, even if it based only on submissions data
        else:
            self.exper_meta_features['users_amount'] = len(submissions_writers.keys())
            if self.exper_meta_features['users_amount'] > 0:
                self.exper_meta_features['deleted_users_normalized'] = \
                    deleted_authors * 1.0 / self.exper_meta_features['users_amount']
            else:
                self.exper_meta_features['deleted_users_normalized'] = 0

    def extract_exper_struct_features(self):
        # creating an object of a SN, then building the netwrok (using networkX and then extracting the features
        csn_obj = CommunitySocialNetwork(name=self.name, submissions=self.exper_submissions_as_list,
                                         comments=self.exper_comments_as_list)
        csn_obj.extract_social_relations(remove_self_loops=True)
        csn_obj.create_nx_graph()
        csn_obj.extract_structural_features()
        # now saving the features
        self.exper_struct_features = csn_obj.structural_features
        return csn_obj.structural_features

    def extract_liwc_features(self, pre_exper=True, while_exper=True, max_texts=-1):
        if pre_exper:
            if self.submissions_as_list is not None:
                text_for_liwc = [esal[1] for esal in self.submissions_as_list]
                text_for_liwc2 = [esal[2] for esal in self.submissions_as_list]
                text_for_liwc3 = [esal[1] for esal in self.comments_as_list]
                text_for_liwc_agg = text_for_liwc + text_for_liwc2 + text_for_liwc3
                text_for_liwc_agg = [tfla for tfla in text_for_liwc_agg if type(tfla) is str]
                # limiting the number of texts to take
                text_for_liwc_agg = text_for_liwc_agg if max_texts == -1 else text_for_liwc_agg[0:max_texts]
                self.liwc_features = extract_liwc_occurrences(text_for_liwc_agg)
            else:
                print(f"liwc_features for sr {self.name} cannot be created for pre_exper since submissions_as_list "
                      f"has not be set yet.")
        if while_exper:
            if self.exper_submissions_as_list is not None:
                text_for_liwc = [esal[1] for esal in self.exper_submissions_as_list]
                text_for_liwc2 = [esal[2] for esal in self.exper_submissions_as_list]
                text_for_liwc3 = [esal[1] for esal in self.exper_comments_as_list]
                text_for_liwc_agg = text_for_liwc + text_for_liwc2 + text_for_liwc3
                text_for_liwc_agg = [tfla for tfla in text_for_liwc_agg if type(tfla) is str]
                # limiting the number of texts to take
                text_for_liwc_agg = text_for_liwc_agg if max_texts == -1 else text_for_liwc_agg[0:max_texts]
                self.exper_liwc_features = extract_liwc_occurrences(text_for_liwc_agg)
            else:
                print(f"liwc_features for sr {self.name} cannot be created for while_exper since "
                      f"exper_submissions_as_list has not be set yet.")

    def get_tiles_placement_inside_paths(self, tiles_placement, pixels_path=None, min_percentage_in_path=0.5):
        if pixels_path is None:
            pixels_path = self.pixels_path
        full_artwork_indices = list()
        for cur_pixels_path in pixels_path:
            if cur_pixels_path['percentage'] < min_percentage_in_path:
                continue
            cur_artwork_indices = get_indices_inside_artwork(cur_pixels_path['path'])
            cur_artwork_indices = pd.DataFrame(cur_artwork_indices, columns=['x_coordinate', 'y_coordinate'])
            full_artwork_indices.append(cur_artwork_indices)
        try:
            full_artwork_indices_df = pd.concat(full_artwork_indices)
        # in case 'full_artwork_indices' is an empty list
        except ValueError:
            full_artwork_indices_df = pd.DataFrame(columns=['x_coordinate', 'y_coordinate'])
        tiles_placement_inside_paths = pd.merge(tiles_placement, full_artwork_indices_df, how='inner',
                                                on=["x_coordinate", "y_coordinate"])
        return tiles_placement_inside_paths

    def update_exper_words_dicts(self, update_only_submissions_dict=False):
        if self.exper_submissions_as_tokens is None or \
                (not update_only_submissions_dict and self.exper_comments_as_tokens is None):
            warnings.warn("An update to the words dictionary was requested to be done, but one of the submission "
                          "or comments (or both) was not initialized. Please set self.exper_submissions_as_tokens "
                          "and/or self.exper_comments_as_tokens before calling this function")
        # case we can really update the dictionary (of submissions)
        if self.exper_submissions_as_tokens is not None:
            self.exper_submissions_tokens_dict = \
                defaultdict(int, Counter([w for sub in self.exper_submissions_as_tokens for sent in sub for w in sent]))
        # case we can really update the dictionary (of comments)
        if not update_only_submissions_dict and self.exper_comments_as_tokens is not None:
            self.exper_comments_tokens_dict = \
                defaultdict(int, Counter([w for sub in self.exper_comments_as_tokens for sent in sub for w in sent]))
        return 0

    def prepare_text_for_bert(self, submissions_sampling=True, sampling_logic='score', sampling_percentage=0.5,
                              max_subm=10000, random_seed=1984):

        if self.submissions_as_list is None and self.exper_submissions_as_list is None:
            print(f"Both 'submissions_as_list' and 'exper_submissions_as_list' are None, the function "
                  f"'prepare_text_to_bert_model' over {self.name} is irrelevant.")
            return 1
        # we can only handle the object, if submissions_as_list has already been created
        if self.submissions_as_list is not None:
            if submissions_sampling:
                # if sampling is required, we do it now (on-the-fly)
                self.subsample_submissions_data(subsample_logic=sampling_logic, percentage=sampling_percentage,
                                                maximum_submissions=max_subm, seed=random_seed)
            cur_sr_sentences = []
            # looping over each submission in the list of submissions
            for idx, i in enumerate(self.submissions_as_list):
                # case both (submission header + body are string)
                if type(i[1]) is str and type(i[2]) is str:
                    cur_sr_sentences.append(i[1] + ' ' + i[2])
                    continue
                # case only the submissions header is a string
                elif type(i[1]) is str:
                    cur_sr_sentences.append(i[1])
                    continue

            # removal of links and replacing them with <link>
            cur_sr_sentences = [RedditDataPrep.mark_urls(sen, marking_method='tag')[0] for sen in cur_sr_sentences]
            self.text_data_for_bert = ' [SEP] '.join(cur_sr_sentences)
        # experiment data is not being filtered in any-case
        if self.exper_submissions_as_list is not None:
            cur_sr_sentences = []
            # looping over each submission in the list of submissions
            for idx, i in enumerate(self.exper_submissions_as_list):
                # case both (submission header + body are string)
                if type(i[1]) is str and type(i[2]) is str:
                    cur_sr_sentences.append(i[1] + ' ' + i[2])
                    continue
                # case only the submissions header is a string
                elif type(i[1]) is str:
                    cur_sr_sentences.append(i[1])
                    continue
            # removal of links and replacing them with <link>
            cur_sr_sentences = [RedditDataPrep.mark_urls(sen, marking_method='tag')[0] for sen in cur_sr_sentences]
            self.exper_text_data_for_bert = ' [SEP] '.join(cur_sr_sentences)

        return 0

    def subsample_submissions_data(self, subsample_logic='score', percentage=0.2, maximum_submissions=5000,
                                   apply_to_tokens=False, seed=1984):
        """
        This function is taken "as-is" from the "SubReddit" class (sr_classifier\sub_reddit.py)
        subsample the submission data of the SR and replaces the list of submissions with the subsample. Logic how to
        do the sub sample is one out of 3 options ('random', 'date', 'score')
        :param subsample_logic: string. Default: 'score'
            the method/logic how to do the sub sampling:'score' means that the top posts based their score will be taken
            'date' means that the latest posts will be taken, 'random' means that random submissions will be taken
        :param percentage: float. Default: 0.2
            the % of submissions to sub sample from the SR. Must be a number > 0 and < 1
        :param maximum_submissions: int. Default: 5000
            maximum number of submissions to sub sample. If the % required to be taken based on 'percentage' input
            yields a larger number than 'maximum_submissions' - than we will sub sample 'maximum_submissions'
        :param apply_to_tokens: bool. Default: False
            whether to apply the sorting and the subseting also to submissions_as_tokens. Such operation might not
            work in case the length of submissions_as_list and submissions_as_tokens is different. In such case, only
            submssions_list will be sorted
        :param seed: int. Default: 1984
            the seed to be used when using 'random' option for sub sampling. Other wise it is not used
        :return: int
            updates the object on the fly (updates the submission_as_list and submissions_as_tokens if required)
            returns 0 in case all went as planned, -1 in case the tokens were not sorted due to length difference
        """
        should_apply_to_tokens = False
        value_to_return = 0
        if apply_to_tokens and len(self.submissions_as_list) == len(self.submissions_as_tokens):
            should_apply_to_tokens = True
        # case we should apply the sorting and sampling also to tokens, but length of the lists differ
        if apply_to_tokens and len(self.submissions_as_list) != len(self.submissions_as_tokens):
            print("SR {} cannot be sorted according by the tokens, "
                  "since length of tokens and submissions differ".format(self.name))
            value_to_return = -1

        # case we want to take the top X% of submissions with the higest score, we'll sort the data according to the it
        if subsample_logic == 'score':
            if should_apply_to_tokens:
                submissions_as_tokens_with_score = [(sat, sal[0]) for sal, sat in
                                                    zip(self.submissions_as_list, self.submissions_as_tokens)]
                submissions_as_tokens_with_score.sort(key=lambda tup: tup[1], reverse=True)
                self.submissions_as_tokens = [sats[0] for sats in submissions_as_tokens_with_score]
                self.submissions_as_list.sort(key=lambda tup: tup[0], reverse=True)
            else:
                self.submissions_as_list.sort(key=lambda tup: tup[0], reverse=True)
        # case we want to take the top X% of submissions latest date, we'll reverse the original list (it is ordered, by
        # the other way round than what is needed)
        elif subsample_logic == 'date':
            if should_apply_to_tokens:
                self.submissions_as_tokens = list(reversed(self.submissions_as_tokens))
                self.submissions_as_list = list(reversed(self.submissions_as_list))
            else:
                self.submissions_as_list = list(reversed(self.submissions_as_list))
        # case we want to randomly choose the submissions, we'll mix all of them and then choose the top X%
        elif subsample_logic == 'random':
            random.seed(seed)
            if should_apply_to_tokens:
                temp_list = list(zip(self.submissions_as_list, self.submissions_as_tokens))
                random.shuffle(temp_list)
                self.submissions_as_list, self.submissions_as_tokens = zip(*temp_list)
            else:
                random.shuffle(self.submissions_as_list)

        # setting the number of submissions to take accrding to the values given as input
        try:
            sub_to_take = math.ceil(len(self.submissions_as_list) * 1.0 * percentage)
        # case there are no submissions at all
        except TypeError:
            return 0
        sub_to_take = sub_to_take if sub_to_take < maximum_submissions else maximum_submissions
        self.submissions_as_list = self.submissions_as_list[0:sub_to_take]
        if should_apply_to_tokens:
            self.submissions_as_tokens = self.submissions_as_tokens[0:sub_to_take]
        return value_to_return

    @staticmethod
    def pixels_cnt_per_user(tiles_placement, user_name, min_ts=1491151733000):
        # cur min_ts is 24 hours back from the end of r/place
        if min_ts is not None:
            tiles_placement_cur_user = \
                tiles_placement[(tiles_placement["ts"] >= min_ts) & (tiles_placement['original_username'] == user_name)]
        # we can choose not to use the 'min_ts' option. In such case, ALL pixels allocation are taken into account
        else:
            tiles_placement_cur_user = tiles_placement[tiles_placement['original_username'] == user_name]
        return tiles_placement_cur_user.shape[0]

    @staticmethod
    def get_max_ts_tiles_placement(tiles_placement):
        # idea is taken from here: https://stackoverflow.com/questions/15705630/get-the-rows-which-have-the-max-value-in-groups-using-groupby
        max_ts_df = tiles_placement.loc[tiles_placement.groupby(["x_coordinate", "y_coordinate"])["ts"].idxmax()]
        return max_ts_df
