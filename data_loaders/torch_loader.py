import re
import os
import random
from torchtext import data
import pickle
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
import datetime
import gc
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from torch.utils.data import DataLoader


#class TorchLoader(data.Dataset):
class TorchLoader(DataLoader):
    cv_splits = None

    def __init__(self, text_field, label_field, sr_name_field, meta_data_field, config_dict, examples=None,
                 verbose=True, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        machine = config_dict['machine']
        self.data_path = config_dict['data_dir'][machine]
        self.sr_objects_path = os.path.join(self.data_path, 'sr_objects', config_dict['srs_obj_file'][machine])
        self.config_dict = config_dict
        #self.cv_splits = None  # this is the splitting of examples into train/test - will be set later

        # now we start loading (this is part of the init!)
        #text_field.preprocessing = data.Pipeline(clean_str)
        start_time = datetime.datetime.now()
        data_path = config_dict['data_dir'][machine]
        fields = [('text', text_field), ('label', label_field),
                  ('sr_name', sr_name_field), ('meta_data', meta_data_field)]
        if examples is None:
            examples = []
            sr_objs = pickle.load(open(self.sr_objects_path, "rb"))
            #sr_objs = sr_objs[0:100]
            # creating tokenized information to each SR, if it doesn't exist
            dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
            min_sent_length = max(config_dict['kernel_sizes'])
            for loop_idx, cur_sr in enumerate(sr_objs):
                if verbose and (loop_idx % 400 == 0) and loop_idx != 0:
                    duration = (datetime.datetime.now() - start_time).seconds
                    print("Finished loading {} objects. Took us up to now: {} sec".format(loop_idx, duration))
                # submission data under sampling
                sampling_dict = config_dict['submissions_sampling']
                if eval(sampling_dict['should_sample']):
                    cur_sr.subsample_submissions_data(subsample_logic=sampling_dict['sampling_logic'],
                                                      percentage=sampling_dict['percentage'],
                                                      maximum_submissions=sampling_dict['max_subm'],
                                                      seed=config_dict['random_seed'])

                full_tok_text = []
                for st in cur_sr.submissions_as_list:
                    # case the self-text is not none (in case it is none, we'll just take the header or the self text)
                    if type(st[2]) is str and type(st[1]) is str:
                        cur_tok_words = dp_obj.tokenize_text(sample=st[1] + '. ' + st[2], convert_to_lemmas=False)
                    elif type(st[1]) is str:
                        cur_tok_words = dp_obj.tokenize_text(sample=st[1], convert_to_lemmas=False)
                    elif type(st[2]) is str:
                        cur_tok_words = dp_obj.tokenize_text(sample=st[2], convert_to_lemmas=False)
                    else:
                        continue
                    full_tok_text.append(cur_tok_words)
                # after going over all submissions, we add it to the object itself
                submission_tokens_as_one_list = []
                for sent in full_tok_text:
                    # filter out sentences which are too short (shorter than minimum kernel size)
                    if len(sent) <= min_sent_length:
                        continue
                    for w in sent:
                        submission_tokens_as_one_list.append(w)
                    submission_tokens_as_one_list.append('<SENT_ENDS>')

                cur_sr.submission_tokens_as_one_list = submission_tokens_as_one_list

                # data prep to the explanatory features (adding the network features/com2vec_algorithm if needed)
                if eval(config_dict['meta_data_usage']['use_network']):
                    net_file_path = os.path.join(data_path, config_dict['meta_data_usage']['network_file_path'][machine])
                else:
                    net_file_path = None
                if eval(config_dict['meta_data_usage']['use_communities_overlap']):
                    com_overlap_file_path = os.path.join(data_path, config_dict['meta_data_usage']['communities_overlap_file_path'][machine])
                else:
                    com_overlap_file_path = None
                cur_sr.meta_features_handler(net_feat_file=net_file_path, com_overlap_file=com_overlap_file_path)

            # case we wish to use explanatory features as part of the modeling
            if eval(config_dict['meta_data_usage']['use_meta']):
                examples += [data.Example.fromlist([sr.submission_tokens_as_one_list, sr.trying_to_draw,
                                                    sr.name, sr.explanatory_features], fields) for sr in sr_objs]
            # case explanatory features are not taking part - it will be None in each example
            else:
                examples += [data.Example.fromlist([sr.submission_tokens_as_one_list, sr.trying_to_draw,
                                                    sr.name, None], fields) for sr in sr_objs]

            del sr_objs
            gc.collect()
        # case cv_splits were not initialized before - we'll do it here for the first time
        if TorchLoader.cv_splits is None:
            cv_obj = StratifiedKFold(n_splits=config_dict['cv']['folds'],
                                     random_state=config_dict['random_seed'])
            y_data = [ex.label for ex in examples]
            TorchLoader.cv_splits = list(cv_obj.split(examples, y_data))
        super(TorchLoader, self).__init__(examples, fields, **kwargs)

    # splitting the data into folds of train/test (usually 5 folds)

    '''
    def init_folds_split(cls, ):
        cv_obj = StratifiedKFold(n_splits=self.config_dict['cv']['folds'], random_state=self.config_dict['random_seed'])
        y_data = [ex.label for ex in self.examples]
        cv_splits = list(cv_obj.split(self.examples, y_data))
    '''
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @staticmethod
    def clean_str(string):
        """
        AVRAHAMI - change this to call spacy tokenizer
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    @classmethod
    def splits(cls, text_field, label_field, sr_name_field, meta_data_field, config_dict, fold_number,
               dev_ratio=.1, dev_shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the subreddits dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # note that although cls is a class call to a constructor, a list of examples is returned
        # (because of the .examples at the end of the construction call)
        examples = cls(text_field=text_field, label_field=label_field, sr_name_field=sr_name_field,
                       meta_data_field=meta_data_field, config_dict=config_dict, **kwargs).examples
        # creating CV splits of the data (usually into 5 folds)

        #cv_obj = StratifiedKFold(n_splits=config_dict['cv']['folds'], random_state=config_dict['random_seed'])
        #y_data = [ex.label for ex in examples]
        #cv_splits = list(cv_obj.split(examples, y_data))
        train_index, test_index = TorchLoader.cv_splits[fold_number]#list(cv_splits)[fold_number]
        cur_train_examples = [examples[i] for i in train_index]
        cur_test_examples = [examples[i] for i in test_index]
        if dev_shuffle:
            random.Random(x=config_dict['random_seed']).shuffle(cur_train_examples)
        cur_dev_index = -1 * int(dev_ratio*len(cur_train_examples))

        cur_train_class = cls(text_field=text_field, label_field=label_field, sr_name_field=sr_name_field,
                              meta_data_field=meta_data_field, config_dict=config_dict,
                              examples=cur_train_examples[:cur_dev_index])
        cur_dev_class = cls(text_field=text_field, label_field=label_field, sr_name_field=sr_name_field,
                            meta_data_field=meta_data_field, config_dict=config_dict,
                            examples=cur_train_examples[cur_dev_index:])
        cur_test_class = cls(text_field=text_field, label_field=label_field, sr_name_field=sr_name_field,
                             meta_data_field=meta_data_field, config_dict=config_dict,
                             examples=cur_test_examples)
        return cur_train_class, cur_dev_class, cur_test_class

