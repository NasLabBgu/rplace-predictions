# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021

from collections import defaultdict
import random
import numpy as np
import pandas as pd
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
import collections
import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from gensim.models import KeyedVectors


class NNClassifier(object):
    """
    Abstract class to work with neural net models, using Dynet

    Parameters
    ----------
    tokenizer: tokenizer obj
        MAYBE IT IS NOT NEEDED
        the object which should tokenize sentences
    eval_measures: dict
        dictionary of evlaution measures to use for measuring the model's performance. Each key is a string of
        the measure name, each value is a function to be used for evaluation (e.g. {'accuracy': accuracy_score})
    emb_size: int. default: 100
        size of the embedding vector. Note it mush be aligned with the size of the matrix being used for embedding
        in case we use external source (i.e., if we use glove pre trained embedding matrix, it must be aligned with it)
    hid_size: int. default: 100
        size of the hidden layer dimension (# of nodes in the hidden layer)
    early_stopping: boolean. default: True
        whether or not to apply early stopping logic along execution. Such early stopping will stop the epochs
        iterations if we passed more than 50% of the iterations and we don't see an improvement in train (!) results
        of more than 1% (between iterations)
    epochs: int. default: 10
        max number of epochs to apply. Eventually, this number can be lowet due to early_stopping logic
    use_meta_features: boolean. default: True
        whether or not to use meta features for modeling
    batch_size: int. default: 10
        number of instances to be included in each learning phase of the network. This is based on the
        implementation of dynet for auto-batching
    seed: int. default: 1984
        the random seed to be used along execution

    Attributes
    ----------
    w2i: defaultdict
        words to integer dictionary
    t2i: defaultdict
        tokens to integer dictioanry
    nwords: int
        number of words in the corpus
    ntags: int
        number of tags in the corpus (in binary classification will be 2)
    eval_results: defaultdict
        evaluation results of the dynet model built
    """

    def __init__(self, model, eval_measures, tokenizer=None, emb_size=100, hid_size=100, early_stopping=True,
                 epochs=10, use_meta_features=True, batch_size=10, seed=1984):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_measures = eval_measures
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.early_stopping = early_stopping
        self.epochs = epochs
        self.use_meta_features = use_meta_features
        self.batch_size = batch_size
        self.seed = seed
        random.seed(self.seed)
        # Functions to read in the corpus
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.t2i = defaultdict(lambda: len(self.t2i))
        self.w2i["<unk>"]
        self.nwords = None
        self.ntags = None
        self.eval_results = defaultdict(list)

    def get_reddit_sentences(self, sr_objects):
        """
        pulls out the sentences related to each sr_object in the list provided
        :param sr_objects: list (of sr_objects)
            list containing sr_objects
        :return: generator
            a generator containing a tuple with 3 variables:
            1. embedded sentences - list of lists, containing in each place the embedded sentences
                (e.g. [[2,4,5], [234, 455, 1]])
            2.  tag - the tag of the sr, in the 1 / 0 format (1 means drawing)
            3. name - string holding the sr name
        """
        # looping over all sr objects
        for cur_sr in sr_objects:
            # pulling out the tag of the current sr
            tag = cur_sr.trying_to_draw
            cur_sr_sentences = []
            # looping over each submission in the list of submissions
            for idx, i in enumerate(cur_sr.submissions_as_list):
                # case both (submission header + body are string)
                if type(i[1]) is str and type(i[2]) is str:
                    cur_sr_sentences.append(i[1] + ' ' + i[2])
                    continue
                # case only the submissions header is a string
                elif type(i[1]) is str:
                    cur_sr_sentences.append(i[1])
                    continue

            # before we tokenize the data, we remove the links and replace them with <link>
            cur_sr_sentences = [RedditDataPrep.mark_urls(sen, marking_method='tag')[0] for sen in cur_sr_sentences]
            cur_sr_sentences_as_int = []
            # converting the words into embeddings and returning the tuple for each sr (note it is a generator)
            for sen in cur_sr_sentences:
                sen_tokenized = self.tokenizer(sen)
                if len(sen_tokenized) > 0:
                    cur_sr_sentences_as_int.append([self.w2i[x] for x in sen_tokenized])
            yield (cur_sr_sentences_as_int, self.t2i[tag], cur_sr.name)

        # updating the # of words and tags we found
        self.nwords = len(self.w2i)
        self.ntags = len(self.t2i)

    @staticmethod
    def data_prep_meta_features(train_data, test_data, update_objects=True):
        """
        function to handle and prepare meta features for the classification model
        :param train_data: list
            list of sr objects to be used as train set
        :param test_data: list
            list of sr objects to be used as train set
        :param update_objects: boolean. default: True
            whether or not to update the sr objects "on the fly" along the function. If False, the objects are not
            updated and the updated dictionaries are returned by the function
        :return: int or dicts
            in case the update_objects=True should just return 0.
            If it is False, 2 dictionaries containing the meta features are returned
        """
        # pulling out the meta features of each object and converting it as pandas df
        train_meta_features = {sr_obj.name: sr_obj.explanatory_features for sr_obj in train_data}
        test_meta_features = {sr_obj.name: sr_obj.explanatory_features for sr_obj in test_data}

        train_as_df = pd.DataFrame.from_dict(train_meta_features, orient='index')
        test_as_df = pd.DataFrame.from_dict(test_meta_features, orient='index')

        # imputation phase
        imp_obj = SimpleImputer(strategy='mean', copy=False)
        train_as_df = pd.DataFrame(imp_obj.fit_transform(train_as_df), columns=train_as_df.columns, index=train_as_df.index)
        test_as_df = pd.DataFrame(imp_obj.transform(test_as_df), columns=test_as_df.columns, index=test_as_df.index)

        # normalization phase
        normalize_obj = StandardScaler()
        train_as_df = pd.DataFrame(normalize_obj.fit_transform(train_as_df), columns=train_as_df.columns, index=train_as_df.index)
        test_as_df = pd.DataFrame(normalize_obj.transform(test_as_df), columns=test_as_df.columns, index=test_as_df.index)

        # creating dicts to be used for replacing the meta features (or return them as is - depends on  the
        # 'update_objects' input param
        train_meta_features_imputed = train_as_df.to_dict('index')
        test_meta_features_imputed = test_as_df.to_dict('index')
        updated_meta_features_dict_train = \
            {key: collections.defaultdict(None, value) for key, value in train_meta_features_imputed.items()}
        updated_meta_features_dict_test = \
            {key: collections.defaultdict(None, value) for key, value in test_meta_features_imputed.items()}
        if update_objects:
            # looping over all srs object and updating the meta data features
            for cur_sr in train_data:
                cur_sr.explanatory_features = updated_meta_features_dict_train[cur_sr.name]
            for cur_sr in test_data:
                cur_sr.explanatory_features = updated_meta_features_dict_test[cur_sr.name]
            return 0
        else:
            return updated_meta_features_dict_train, updated_meta_features_dict_test

    def build_embedding_matrix(self, embedding_file, add_extra_row=True):
        """
        building an embedding matrix based on a given external file. Such matrix is a combination of words we
        identify in the exteranl file and words that do not appear there and will be initialize with a random
        embedding vector
        :param embedding_file: str
            the path to the exact embedding file to be used. This should be a txt file, each row represents
            a word and it's embedding (separated by whitespace). Example can be taken from 'glove' pre-trained models
        :return: numpy matrix
            the embedding matrix built
        """
        start_time = datetime.datetime.now()
        # building the full embedding matrix, with random normal values. Later we'll replace known words
        if add_extra_row:
            embedding_matrix = np.random.normal(loc=0.0, scale=1.0, size=(self.nwords + 1, self.emb_size))
        else:
            embedding_matrix = np.random.normal(loc=0.0, scale=1.0, size=(self.nwords, self.emb_size))
        embeddings_index = dict()
        found_words = 0

        # passing over the pretrained embedding matrix
        if embedding_file.endswith('.txt'):
            with open(embedding_file) as infile:
                for idx, line in enumerate(infile):
                    values = line.split()
                    word = values[0]
                    try:
                        coefs = np.asarray(values[1:], dtype='float32')
                    except ValueError:
                        continue
                    embeddings_index[word] = coefs
                    # updating the embedding matrix according to the w2i dictionary (if is it found)
                    if word in self.w2i:
                        embedding_matrix[self.w2i[word]] = coefs
                        found_words += 1
                infile.close()

        elif embedding_file.endswith('.vec'):
            w2v_model = KeyedVectors.load_word2vec_format(embedding_file)
            models_dict = set(w2v_model.vocab.keys())
            for idx, word in enumerate(models_dict):
                values = w2v_model[word]
                try:
                    coefs = np.asarray(values, dtype='float32')
                except ValueError:
                    continue
                embeddings_index[word] = coefs
                # updating the embedding matrix according to the w2i dictionary (if is it found)
                if word in self.w2i:
                    embedding_matrix[self.w2i[word]] = coefs
                    found_words += 1

        else:
            raise IOError("Embedding file format not recognized")
        duration = (datetime.datetime.now() - start_time).seconds
        print("We have finished running the 'build_embedding_matrix' function. Took us {0:.2f} seconds. "
              "We have found {1:.1f}% of matching words "
              "compared to the embedding matrix.".format(duration, found_words * 100.0 / self.nwords))
        return embedding_matrix

    def calc_eval_measures(self, y_true, y_pred, nomalize_y=True):
        """
        calculation of the evaluation measures for a given prediciton vector and the y_true vector
        :param y_true: list of ints
            list containing the true values of y. Any value > 0 is considered as 1 (drawing),
            all others are 0 (not drawing)
        :param y_pred: list of floats
            list containing prediction values for each sr. It represnts the probability of the sr to be a drawing one
        :param nomalize_y: boolean. default: True
            whether or not to normalize the y_true and the predictions
        :return: dict
            dictionary with all the evalution measures calculated
        """
        if nomalize_y:
            y_true = [1 if y > 0 else 0 for y in y_true]
            binary_y_pred = [1 if p > 0.5 else 0 for p in y_pred]
        else:
            binary_y_pred = [1 if p > 0.5 else -1 for p in y_pred]
        for name, func in self.eval_measures.items():
            self.eval_results[name].append(func(y_true, binary_y_pred))
        return self.eval_results

    @staticmethod
    def from_spec(spec, model):
        raise NotImplementedError
