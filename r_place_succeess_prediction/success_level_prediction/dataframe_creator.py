import os
from os.path import join as opj
import pickle
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import multiprocessing as mp
import datetime
from collections.abc import MutableMapping
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class DataFrameCreator(object):
    def __init__(self, target_feature, path_to_srs_obj, path_to_target_df, while_or_before_exper,
                 meta_features, network_features, liwc, bow, doc2vec, com2vec, snap, graph2vec,
                 maximum_srs=None, debug_mode=False, bow_params={'max_df': 0.8, 'min_df': 3, 'max_features': 300}):
        self.target_feature = target_feature
        self.path_to_srs_obj = path_to_srs_obj
        self.path_to_target_df = path_to_target_df
        self.while_or_before_exper = while_or_before_exper
        self.use_meta_features = meta_features
        self.use_network_features = network_features
        self.use_liwc = liwc
        self.use_bow = bow
        self.use_doc2vec = doc2vec
        self.use_com2vec = com2vec
        self.use_snap = snap
        self.use_graph2vec = graph2vec
        self.com_names = self.extract_com_names(maximum_srs=maximum_srs)
        self.y_feature = None
        self.debug_mode = debug_mode
        self.bow_params = bow_params

    def extract_com_names(self, maximum_srs=None):
        com_level_gold_label_df = pickle.load(open(self.path_to_target_df, "rb"))
        # taking only srs which were manually labeled (~300 srs exist but did not really participate in r/place)
        com_level_gold_label_df = com_level_gold_label_df[com_level_gold_label_df['manually_labeled']].copy()
        if self.while_or_before_exper == 'before':
            sr_names = list(com_level_gold_label_df[com_level_gold_label_df['is_created_for_rplace'] == False].index)
        else:
            sr_names = list(com_level_gold_label_df.index)

        sr_names = sorted(sr_names)
        # filtering some SRs - if required
        if maximum_srs is not None:
            sr_names = sr_names[0:maximum_srs]
        return sr_names

    @staticmethod
    def extract_doc2vec_embed(file_path, com_names):
        doc2vec_dict = pickle.load(open(file_path, "rb"))
        doc2vec_r_place_com = {sr_name: {'d2v_' + str(feature): number for feature, number in cur_dict.items()}
                               for sr_name, cur_dict in doc2vec_dict.items() if sr_name in com_names}
        return doc2vec_r_place_com

    @staticmethod
    def extract_com2vec_embed(file_path, com_names):
        com2vec_dict = pickle.load(open(file_path, "rb"))
        com2vec_r_place_com = {sr_name: {'c2v_' + str(feature): number for feature, number in cur_dict.items()}
                               for sr_name, cur_dict in com2vec_dict.items() if sr_name in com_names}
        return com2vec_r_place_com

    @staticmethod
    def extract_snap_embed(file_path, com_names):
        snap_embeddings = pd.read_csv(file_path, header=None)
        snap_embeddings.set_index(0, inplace=True, drop=True)
        found_idx = [idx for idx in snap_embeddings.index if idx in com_names]
        snap_embeddings_r_place_com = snap_embeddings.loc[found_idx].to_dict(orient='index')
        snap_embeddings_r_place_com = {sr_name: {'s2v_' + str(feature): number for feature, number in cur_dict.items()}
                                       for sr_name, cur_dict in snap_embeddings_r_place_com.items()}
        return snap_embeddings_r_place_com

    @staticmethod
    def extract_graph2vec_embed(file_path, com_names):
        graph2vec_dict = pickle.load(open(file_path, "rb"))
        graph2vec_r_place_com = {sr_name: {'g2v_' + str(idx): number for idx, number in enumerate(cur_list)}
                                 for sr_name, cur_list in graph2vec_dict.items() if sr_name in com_names}
        return graph2vec_r_place_com

    @staticmethod
    def rec_merge(d1, d2):
        """
        Update two dicts of dicts recursively, if either mapping has leaves that are non-dicts,
        the second's leaf overwrites the first's.
        Code idea is taken from: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
        """
        for k, v in d1.items():
            if k in d2:
                # this next check is the only difference!
                if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                    d2[k] = DataFrameCreator.rec_merge(v, d2[k])
                # we could further check types and merge as appropriate here.
        d3 = d1.copy()
        d3.update(d2)
        return d3

    def extract_features_specific_sr_obj(self, idx, sr_obj_path, verbose=True):
        """
        extracting meta/network/liwc features for a specific community
        features are extracted per definition of the class - before or while the exper
        :param sr_obj_path: str
            path to the object of the subreddit
        :param verbose: bool. Default: False
            whether to print information through the run or not
        :return: dict
            dictionary with all the extracted features. Features which are not found are not added at all
        """
        start_time = datetime.datetime.now()
        if verbose:
            print(f"Extraction of features using the file {sr_obj_path} started!", flush=True)
        features_dict = dict()
        cur_sr_obj = pickle.load(open(sr_obj_path, "rb"))
        sr_name = cur_sr_obj.name
        if self.use_meta_features:
            if self.while_or_before_exper == 'before' and cur_sr_obj.meta_features[0] is not None:
                features_dict.update({key: value for key, value in cur_sr_obj.meta_features[0].items()})
            elif self.while_or_before_exper == 'while' and cur_sr_obj.exper_meta_features is not None:
                features_dict.update({'exper_' + key: value for key, value in cur_sr_obj.exper_meta_features.items()})
        if self.use_network_features:
            if self.while_or_before_exper == 'before' and cur_sr_obj.struct_features is not None:
                # the structural features are a bit complex, so considering here all options
                cur_structural_features_dict = {key: value for key, value in
                                                cur_sr_obj.struct_features.items()
                                                if type(value) is not dict and type(value) is not defaultdict}
                cur_structural_features_dict.update(
                    {key + '_avg': value['avg'] for key, value in cur_sr_obj.struct_features.items()
                     if type(value) is dict})
                cur_structural_features_dict.update(
                    {key + '_num': value['num'] for key, value in cur_sr_obj.struct_features.items()
                     if type(value) is defaultdict})
                features_dict.update(cur_structural_features_dict)
            elif self.while_or_before_exper == 'while' and cur_sr_obj.exper_struct_features is not None:
                cur_structural_features_dict = {'exper_' + key: value for key, value in
                                                cur_sr_obj.exper_struct_features.items()
                                                if type(value) is not dict and type(value) is not defaultdict}
                cur_structural_features_dict.update(
                    {'exper_' + key + '_avg': value['avg'] for key, value in cur_sr_obj.exper_struct_features.items()
                     if type(value) is dict})
                cur_structural_features_dict.update(
                    {'exper_' + key + '_num': value['num'] for key, value in cur_sr_obj.exper_struct_features.items()
                     if type(value) is defaultdict})
                features_dict.update(cur_structural_features_dict)
        if self.use_liwc:
            if self.while_or_before_exper == 'before' and cur_sr_obj.liwc_features is not None:
                tot_words_cnt = sum(cur_sr_obj.liwc_features.values())
                cur_liwc_features = {'liwc_' + key: value / tot_words_cnt
                                     for key, value in cur_sr_obj.liwc_features.items()}
                features_dict.update(cur_liwc_features)
            elif self.while_or_before_exper == 'while' and cur_sr_obj.exper_liwc_features is not None:
                tot_words_cnt = sum(cur_sr_obj.exper_liwc_features.values())
                cur_liwc_features = {'liwc_' + key: value / tot_words_cnt
                                     for key, value in cur_sr_obj.exper_liwc_features.items()}
                features_dict.update(cur_liwc_features)

        duration = (datetime.datetime.now() - start_time).seconds
        if verbose:
            print(f"Extraction of features for sr {sr_name} has ended in {duration} seconds", flush=True)
        return sr_name, features_dict

    def extract_features_from_all_sr_objs(self, verbose=True):
        """
        extracting meta/network/liwc features for a all communities in the corpus
        features are extracted per definition of the class - before or while the exper
        :return: dict
            dictionary with name as the key of the community and value as dictionary with all values
        """
        sr_obj_f_names = sorted([f for f in os.listdir(self.path_to_srs_obj) if re.match(r'.*sr_obj_.*\.p', f)])
        # taking only file names that exists in the list of sr_names of the object
        sr_obj_f_names = [sofn for sofn in sr_obj_f_names if sofn.split('sr_obj_')[-1].split('.p')[0] in self.com_names]
        sr_obj_paths = [opj(self.path_to_srs_obj, sofn) for sofn in sr_obj_f_names]
        data_for_poll = [(idx, sop, verbose) for idx, sop in enumerate(sr_obj_paths)]
        if self.debug_mode:
            pool = mp.Pool(processes=1)  # useful for debugging
        else:
            pool = mp.Pool(processes=int(mp.cpu_count() * 0.75 + 1))
        with pool as pool:
            results = pool.starmap(self.extract_features_specific_sr_obj, data_for_poll)
        results_as_dict = {r[0]: r[1] for r in results}
        return results_as_dict

    def extract_bow_specific_sr_obj(self, idx, sr_obj_path, verbose=False):
        """
        extracting bag-of-words (BOW) features for a specific community
        features are extracted per definition of the class - before or while the exper
        :param sr_obj_path: str
            path to the object of the subreddit
        :param verbose: bool. Default: False
            whether to print information through the run or not
        :return: dict
            dictionary with all the extracted features. Features which are not found are not added at all
        """
        start_time = datetime.datetime.now()
        cur_sr_obj = pickle.load(open(sr_obj_path, "rb"))
        sr_name = cur_sr_obj.name
        if self.while_or_before_exper == 'before':
            tokens_dict_subm = cur_sr_obj.submissions_tokens_dict.copy()
            tokens_dict_comm = cur_sr_obj.comments_tokens_dict.copy()
            # merging two dicts together (of the submissions and comments
            for k, v in tokens_dict_subm.items():
                if k in tokens_dict_comm:
                    tokens_dict_comm[k] += tokens_dict_subm[k]
                else:
                    tokens_dict_comm[k] = tokens_dict_subm[k]
        else:
            tokens_dict_subm = cur_sr_obj.exper_submissions_tokens_dict.copy()
            tokens_dict_comm = cur_sr_obj.exper_comments_tokens_dict.copy()
            # merging two dicts together (of the submissions and comments
            for k, v in tokens_dict_subm.items():
                if k in tokens_dict_comm:
                    tokens_dict_comm[k] += tokens_dict_subm[k]
                else:
                    tokens_dict_comm[k] = tokens_dict_subm[k]
        # now we can use the tokens_dict_comm as the merged dict (either before or while the exper)
        if verbose:
            duration = (datetime.datetime.now() - start_time).seconds
            print(f"Extraction of BOW features for sr {sr_name} has ended in {duration} seconds", flush=True)
        return sr_name, tokens_dict_comm

    def extract_bow_from_all_sr_objs(self, max_df=0.8, min_df=0.3, max_features=100):
        """
        extracting bag-of-words (BOW) features for all communities in the corpus
        features are extracted per definition of the class - before or while the exper
        :return: dict
            dictionary with name as the key of the community and value as dictionary with all values
        """
        sr_obj_f_names = sorted([f for f in os.listdir(self.path_to_srs_obj) if re.match(r'.*sr_obj_.*\.p', f)])
        # taking only file names that exists in the list of sr_names of the object
        sr_obj_f_names = [sofn for sofn in sr_obj_f_names if sofn.split('sr_obj_')[-1].split('.p')[0] in self.com_names]
        sr_obj_paths = [opj(self.path_to_srs_obj, sofn) for sofn in sr_obj_f_names]
        verbose_in_mp = True
        data_for_poll = [(idx, sop, verbose_in_mp) for idx, sop in enumerate(sr_obj_paths)]
        if self.debug_mode:
            pool = mp.Pool(processes=1)  # useful for debugging
        else:
            pool = mp.Pool(processes=int(mp.cpu_count() * 0.75 + 1))
        with pool as pool:
            results = pool.starmap(self.extract_bow_specific_sr_obj, data_for_poll)
        # now the results hold a tuple with the name of the SR in first place and dict of tokens in the second
        results_com_names = list()
        results_list_of_text = list()
        # looping over the results and: adding the name of the sr to the list of SRs + adding long text per each SR
        # to the results_list_of_text (len(results_list_of_text) == len(results_com_names))
        for n, bow_dict in results:
            results_com_names.append(n)
            results_list_of_text.append(' '.join([' '.join([key] * cnt) for key, cnt in bow_dict.items()]))
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
        bow_x_matrix = vectorizer.fit_transform(results_list_of_text)
        f_names = ['bow_'+f_name for f_name in vectorizer.get_feature_names()]
        bow_x_df = pd.DataFrame(bow_x_matrix.toarray(), columns=f_names, index=results_com_names)
        bow_x_dict = bow_x_df.to_dict(orient='index')
        return bow_x_dict

    def extract_text_for_bert_specific_sr_obj(self, idx, sr_obj_path, verbose=False):
        start_time = datetime.datetime.now()
        cur_sr_obj = pickle.load(open(sr_obj_path, "rb"))
        sr_name = cur_sr_obj.name
        if self.while_or_before_exper == 'before':
            bert_text = cur_sr_obj.text_data_for_bert
        elif self.while_or_before_exper == 'while':
            bert_text = cur_sr_obj.exper_text_data_for_bert
        elif self.while_or_before_exper == 'both':
            bert_text1 = cur_sr_obj.text_data_for_bert
            bert_text2 = cur_sr_obj.exper_text_data_for_bert
            bert_text = bert_text2 + ' [SEP] ' + bert_text1
        else:
            raise IOError("Invalid 'while_or_before_exper' parameter. Has to be one out of:('before', 'while', 'both).")
        if verbose:
            duration = (datetime.datetime.now() - start_time).seconds
            print(f"Extraction of BERT text data for sr {sr_name} has ended in {duration} seconds", flush=True)
        return sr_name, bert_text

    def extract_text_for_bert_from_all_sr_objs(self):
        sr_obj_f_names = sorted([f for f in os.listdir(self.path_to_srs_obj) if re.match(r'.*sr_obj_.*\.p', f)])
        # taking only file names that exists in the list of sr_names of the object
        sr_obj_f_names = [sofn for sofn in sr_obj_f_names if sofn.split('sr_obj_')[-1].split('.p')[0] in self.com_names]
        sr_obj_paths = [opj(self.path_to_srs_obj, sofn) for sofn in sr_obj_f_names]
        verbose_in_mp = False
        data_for_poll = [(idx, sop, verbose_in_mp) for idx, sop in enumerate(sr_obj_paths)]
        if self.debug_mode:
            pool = mp.Pool(processes=1)  # useful for debugging
        else:
            pool = mp.Pool(processes=int(mp.cpu_count() * 0.75 + 1))
        with pool as pool:
            results = pool.starmap(self.extract_text_for_bert_specific_sr_obj, data_for_poll)
        results_as_dict = {r[0]: r[1] for r in results}
        return results_as_dict

    def create_y_feature(self, use_percentile=True, normalize=False, log_scale=False):
        """
        Creation of the target feature (y)
        :param use_percentile: boolean. Default: True
            whether to use the percentile of the feature we normalize by (e.g., community size) or the log. If set to
            False, the log (base 10) will be used)
        :param normalize: boolean. Default: False
            Whether to normalize the feature after all modifications or now
        :param log_scale: boolean. Default: False
            Whether to use a log scale of the feature or not
        :return: vector of the y vector
        """
        com_level_gold_label_df = pickle.load(open(self.path_to_target_df, "rb"))
        # making sure no missing value in the 'tot_pixels' columns
        if [tp for tp in com_level_gold_label_df['tot_pixels'] if tp is None or np.isnan(tp)]:
            raise IOError("Error with the 'tot_pixels' column - a missing/None value was detected. Check it please.")
        # in case the target_feature is not the binary one, we are in a case of a regression problme. In such a case
        # we wish to REMOVE all communities with a value of 0 in the tot_pixels
        if self.target_feature != 'binary':
            com_level_gold_label_df = com_level_gold_label_df[com_level_gold_label_df['tot_pixels'] != 0].copy()
        # creating the actual target feature - binary label (survived/did not survive)
        if self.target_feature == 'binary':
            target_feature_dict = \
                {idx: 0 if row['tot_pixels'] < 5 else 1 for idx, row in com_level_gold_label_df.iterrows()}
        elif self.target_feature == 'pixels':
            target_feature_dict = {idx: row['tot_pixels'] for idx, row in com_level_gold_label_df.iterrows()}
        elif self.target_feature == 'pixels_community_size_normalized':
            # this is the first and straight forward option - using the percentile of the factorized target feature
            if use_percentile:
                # create a new column of percentile rank. We use the max function to make sure factorized not overkilled
                com_level_gold_label_df['community_size_percentile'] = com_level_gold_label_df['community_size'].rank(pct=True)
                com_level_gold_label_df['community_size_percentile'] = com_level_gold_label_df['community_size_percentile'].apply(lambda row: max(row, 0.1))
                target_feature_dict = {idx: row['tot_pixels'] * (1 - row['community_size_percentile'])
                                       for idx, row in com_level_gold_label_df.iterrows()}
            # this is the second option, suggested by Oren - to use the log of the factorized feature
            else:
                com_level_gold_label_df['log_community_size'] = [np.log10(cur_y) if cur_y > 1 else 0.01
                                                                 for cur_y in com_level_gold_label_df['community_size']]
                com_level_gold_label_df['log_tot_pixels'] = [np.log10(cur_y) if cur_y != 0 else cur_y
                                                             for cur_y in com_level_gold_label_df['tot_pixels']]
                target_feature_dict = {idx: row['log_tot_pixels'] / row['log_community_size']
                                       for idx, row in com_level_gold_label_df.iterrows()}
        elif self.target_feature == 'pixels_demand_normalized':
            # this is the first and straight forward option - using the percentile of the factorized target feature
            if use_percentile:
                # create a new column of percentile rank. We use the max function to make sure factorized not overkilled
                com_level_gold_label_df['demand_percentile'] = com_level_gold_label_df['mean_demand'].rank(pct=True)
                com_level_gold_label_df['demand_percentile'] = com_level_gold_label_df['demand_percentile'].apply(lambda row: max(row, 0.1))
                target_feature_dict = \
                    {idx: row['tot_pixels'] * row['demand_percentile'] for idx, row in com_level_gold_label_df.iterrows()}
            # this is the second option, suggested by Oren - to use the log of the factorized feature
            else:
                com_level_gold_label_df['log_pixels_demand'] = [np.log10(cur_y) if cur_y != 0 else 0.01 #MUST EDIT HERE!!
                                                                for cur_y in com_level_gold_label_df['mean_demand']]
                com_level_gold_label_df['log_tot_pixels'] = [np.log10(cur_y) if cur_y != 0 else cur_y
                                                             for cur_y in com_level_gold_label_df['tot_pixels']]
                target_feature_dict = {idx: row['log_tot_pixels'] * row['log_pixels_demand']
                                       for idx, row in com_level_gold_label_df.iterrows()}
        elif self.target_feature == 'pixels_entropy_normalized':
            # this is the first and straight forward option - using the percentile of the factorized target feature
            if use_percentile:
                # create a new column of percentile rank. We use the max function to make sure factorized not overkilled
                com_level_gold_label_df['entropy_percentile'] = com_level_gold_label_df['mean_entropy'].rank(pct=True)
                com_level_gold_label_df['entropy_percentile'] = com_level_gold_label_df['entropy_percentile'].apply(lambda row: max(row, 0.1))
                target_feature_dict = \
                    {idx: row['tot_pixels'] * row['entropy_percentile'] for idx, row in com_level_gold_label_df.iterrows()}
            # this is the second option, in the entropy measure, we do not use log since numbers are very low anyway
            else:
                com_level_gold_label_df['log_tot_pixels'] = [np.log10(cur_y) if cur_y != 0 else cur_y
                                                             for cur_y in com_level_gold_label_df['tot_pixels']]
                target_feature_dict = {idx: row['log_tot_pixels'] * row['mean_entropy']
                                       for idx, row in com_level_gold_label_df.iterrows()}
        elif self.target_feature == 'pixels_diameter_normalized':
            # this is the first and straight forward option - using the percentile of the factorized target feature
            if use_percentile:
                # create a new column of percentile rank. We use the max function to make sure factorized not overkilled
                com_level_gold_label_df['diameter_percentile'] = com_level_gold_label_df['tot_diameter'].rank(pct=True)
                com_level_gold_label_df['diameter_percentile'] = com_level_gold_label_df['diameter_percentile'].apply(lambda row: max(row, 0.1))
                target_feature_dict = \
                    {idx: row['tot_pixels'] * row['diameter_percentile'] for idx, row in com_level_gold_label_df.iterrows()}
            else:
                com_level_gold_label_df['log_diameter'] = [np.log10(cur_y) if cur_y != 0 else cur_y
                                                           for cur_y in com_level_gold_label_df['tot_diameter']]
                com_level_gold_label_df['log_tot_pixels'] = [np.log10(cur_y) if cur_y != 0 else cur_y
                                                             for cur_y in com_level_gold_label_df['tot_pixels']]
                target_feature_dict = {idx: row['log_tot_pixels'] * row['log_diameter']
                                       for idx, row in com_level_gold_label_df.iterrows()}
        # the big if (this is in case the target feature is something we do not recognize
        else:
            raise IOError("Error with the 'target_feature' name - has to be one out of six options "
                          "(binary / pixels / pixels_community_size_normalized / "
                          "pixels_demand_normalized / pixels_entropy_normalized / pixels_diameter_normalized.")
        # normalization of y (if required)
        if normalize:
            scaler = StandardScaler()
            normalized_y = scaler.fit_transform(X=np.array(pd.Series(target_feature_dict)).reshape(-1, 1))
            target_feature_dict = {com_name: cur_y for com_name, cur_y in zip(target_feature_dict.keys(), normalized_y[:, 0])}
        # we use log scale only in case use_percentile=True, since if it set to False, we already use log-scale when
        # creating the feature (see the inner if-else blocks)
        if (log_scale and use_percentile) or (log_scale and self.target_feature == 'pixels'):
            target_feature_dict = {com_name: np.log10(cur_y) if cur_y != 0 else cur_y for com_name, cur_y in target_feature_dict.items()}
        return target_feature_dict

    def create_x_matrix(self, verbose=True):
        list_of_extracted_features = list()
        # extracting the BOW features (if required)
        if self.use_bow:
            bow_features = self.extract_bow_from_all_sr_objs(**self.bow_params)
            list_of_extracted_features.append(bow_features)
        # handling of whether to take or not each type (e.g., meta) is decided inside the function below
        meta_struc_liwc_features = self.extract_features_from_all_sr_objs(verbose=verbose)
        list_of_extracted_features.append(meta_struc_liwc_features)
        if self.use_doc2vec:
            if self.while_or_before_exper == 'before':
                model_path = opj('/sise/Yalla_work/data/reddit_place/embedding/embedding_per_sr/doc2vec/3.00/',
                                 'communities_doc2vec_model_5_11_2019_dict.p')
            else:
                model_path = opj('/sise/Yalla_work/data/reddit_place/embedding/embedding_per_sr/doc2vec/',
                                 'doc2vec_dict_while_exper.p')
            doc2vec_r_place_com = self.extract_doc2vec_embed(file_path=model_path, com_names=self.com_names)
            list_of_extracted_features.append(doc2vec_r_place_com)
        if self.use_com2vec:
            if self.while_or_before_exper == 'before':
                model_path = opj('/sise/Yalla_work/data/reddit_place/com2vec_algorithm/com2vec_2015_to_2019_13_4_2019_dict.p')
                com2vec_r_place_com = self.extract_com2vec_embed(file_path=model_path, com_names=self.com_names)
                list_of_extracted_features.append(com2vec_r_place_com)
            else:
                print(f"'use_com2vec' was set to True together with 'while_or_before_exper' set to 'while' - it does"
                      f" not make sense.", flush=True)
        if self.use_snap:
            model_path = opj('/sise/Yalla_work/data/reddit_place/embedding/embedding_per_sr/SNAP/',
                             'web-redditEmbeddings-subreddits.csv')
            snap_embed_r_place_com = self.extract_snap_embed(file_path=model_path, com_names=self.com_names)
            list_of_extracted_features.append(snap_embed_r_place_com)
        if self.use_graph2vec:
            if self.while_or_before_exper == 'before':
                model_path = opj('/sise/Yalla_work/data/reddit_place/embedding/embedding_per_sr/graph2vec/',
                                 'graph2vec_dict_before_exper.p')
            else:
                model_path = opj('/sise/Yalla_work/data/reddit_place/embedding/embedding_per_sr/graph2vec/',
                                 'graph2vec_dict_while_exper.p')
            graph2vec_r_place_com = self.extract_graph2vec_embed(file_path=model_path, com_names=self.com_names)
            list_of_extracted_features.append(graph2vec_r_place_com)
        full_features_per_sr_dict = reduce(DataFrameCreator.rec_merge, tuple(list_of_extracted_features))
        full_features_per_sr_df = pd.DataFrame.from_dict(full_features_per_sr_dict, orient='index')
        # in case the df is empty(cases when all flags are False) - we will create a df with sr names only
        if full_features_per_sr_df.shape[0] == 0:
            full_features_per_sr_df = pd.DataFrame(index=self.com_names)
        return full_features_per_sr_df
