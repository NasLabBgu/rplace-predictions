import numpy as np
import datetime
from gensim.models import KeyedVectors
from sr_classifier.reddit_data_preprocessing import RedditDataPrep
import torch
from torch.nn import ConstantPad1d
import gc
import pandas as pd
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import data_loaders.torch_loader as torch_loader
import torchtext.data as data
import random
import os


def build_embedding_matrix(embedding_file, text_field, emb_size):
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
        nwords = len(text_field.vocab.itos)
        # building the full embedding matrix, with random normal values. Later we'll replace known words
        embedding_matrix = np.random.normal(loc=0.0, scale=1.0, size=(nwords, emb_size))
        embeddings_index = dict()
        found_words = 0

        # passing over the pretrained embedding matrix
        if embedding_file.endswith('.txt'):
            with open(embedding_file, encoding='utf-8') as infile:
                for idx, line in enumerate(infile):
                    values = line.split()
                    word = values[0]
                    try:
                        coefs = np.asarray(values[1:], dtype='float32')
                    except ValueError:
                        continue
                    embeddings_index[word] = coefs
                    # updating the embedding matrix according to the w2i dictionary (if is it found)
                    if word in text_field.vocab.stoi:
                        embedding_matrix[text_field.vocab.stoi[word]] = coefs
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
                if word in text_field.vocab.stoi:
                    embedding_matrix[text_field.vocab.stoi[word]] = coefs
                    found_words += 1

        else:
            raise IOError("Embedding file format not recognized")
        duration = (datetime.datetime.now() - start_time).seconds
        print("We have finished running the 'build_embedding_matrix' function. Took us {0:.2f} seconds. "
              "We have found {1:.1f}% of matching words "
              "compared to the embedding matrix.".format(duration, found_words * 100.0 / nwords))
        return embedding_matrix


def pull_out_text_data(sr_objects):
    """
    pulls out the sentences related to each sr_object in the list provided
    :param sr_objects: list (of sr_objects)
        list containing sr_objects
    :return: list of lists
        ???
    """
    # looping over all sr objects
    for cur_sr in sr_objects:
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
        return [RedditDataPrep.mark_urls(sen, marking_method='tag')[0] for sen in cur_sr_sentences]


def submissions_separation(input_tensor, separator_int, padding_int=1):
    rel_idx = [x[0] for x in (input_tensor[0] == separator_int).nonzero().tolist()]
    rel_idx_len = len(rel_idx)
    # special case if the input tensor includes a single case
    if rel_idx_len == 1:
        return torch.narrow(input_tensor, 1, 0, rel_idx[0]-1)
    sent_length = [j-i-1 for i, j in zip(rel_idx[:-1], rel_idx[1:])]
    new_tensors_list = []
    max_sent_len = max(sent_length)
    # special case for the first appearance
    new_tensor = torch.narrow(input_tensor, 1, 0, rel_idx[0])
    padding_obj = ConstantPad1d((0, max_sent_len - len(new_tensor[0])), padding_int)
    new_tensor = padding_obj(new_tensor)
    new_tensors_list.append(new_tensor)
    if rel_idx_len > 1:
        for loop_num, (i, sl) in enumerate(zip(range(rel_idx_len-1), sent_length)):
            new_tensor = torch.narrow(input_tensor, 1, rel_idx[i] + 1, sl)
            padding_obj = ConstantPad1d((0, max_sent_len - len(new_tensor[0])), padding_int)
            new_tensor = padding_obj(new_tensor)
            new_tensors_list.append(new_tensor)
        # special case for the last case - not needed since we should have an ending sign @ the end of the last sentence
    return torch.cat(new_tensors_list)


def data_prep_meta_features(*args, train_data, fill_missing_values=True, update_objects=True):
    """
    AVRAHAMI-FIX THE DOCUMENTATION HERE
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
    test_meta_features = []
    train_meta_features = {example.sr_name: example.meta_data for example in train_data.examples}
    for arg in args:
        test_meta_features.append({example.sr_name: example.meta_data for example in arg.examples})

    test_as_df = []
    train_as_df = pd.DataFrame.from_dict(train_meta_features, orient='index')
    for cur_test_mf in test_meta_features:
        test_as_df.append(pd.DataFrame.from_dict(cur_test_mf, orient='index'))

    # imputation phase
    imp_obj = SimpleImputer(strategy='mean', copy=False)
    train_as_df = pd.DataFrame(imp_obj.fit_transform(train_as_df), columns=train_as_df.columns, index=train_as_df.index)
    for idx, cur_test_as_df in enumerate(test_as_df):
        test_as_df[idx] = pd.DataFrame(imp_obj.transform(test_as_df[idx]), columns=test_as_df[idx].columns,
                                       index=test_as_df[idx].index)

    # normalization phase (only doing it to the ones which are not related to the com2vec_algorithm ones)
    columns_to_transpose = [col for col in train_as_df.columns if not col.startswith('com_overlap')]
    columns_not_to_transpose = [col for col in train_as_df.columns if col.startswith('com_overlap')]
    normalize_obj = StandardScaler()
    train_as_array = np.concatenate((normalize_obj.fit_transform(train_as_df[columns_to_transpose]),
                                     train_as_df[columns_not_to_transpose]), axis=1)
    train_as_df = pd.DataFrame(train_as_array, columns=train_as_df.columns, index=train_as_df.index)
    for idx, cur_test_as_df in enumerate(test_as_df):
        test_as_array = np.concatenate((normalize_obj.fit_transform(test_as_df[idx][columns_to_transpose]),
                                        test_as_df[idx][columns_not_to_transpose]), axis=1)

        test_as_df[idx] = pd.DataFrame(test_as_array, columns=test_as_df[idx].columns, index=test_as_df[idx].index)

    if fill_missing_values:
        train_as_df.fillna(value=0.0, inplace=True)
        for idx, cur_test_as_df in enumerate(test_as_df):
            test_as_df[idx].fillna(value=0.0, inplace=True)
    # creating dicts to be used for replacing the meta features (or return them as is - depends on  the
    # 'update_objects' input param
    train_meta_features_imputed = train_as_df.to_dict('index')
    test_meta_features_imputed = []
    for idx, cur_test_as_df in enumerate(test_as_df):
        test_meta_features_imputed.append(cur_test_as_df.to_dict('index'))

    updated_meta_features_dict_test = []
    updated_meta_features_dict_train = \
        {key: collections.defaultdict(None, value) for key, value in train_meta_features_imputed.items()}
    for idx, cur_test_meta_features_imputed in enumerate(test_meta_features_imputed):
        updated_meta_features_dict_test.append({key: collections.defaultdict(None, value)
                                                for key, value in cur_test_meta_features_imputed.items()})
    if update_objects:
        # looping over all srs object and updating the meta data features
        for cur_example in train_data:
            cur_example.meta_data = updated_meta_features_dict_train[cur_example.sr_name]
        for idx, arg in enumerate(args):
            for cur_example in arg:
                cur_example.meta_data = updated_meta_features_dict_test[idx][cur_example.sr_name]
        return 0
    else:
        return (updated_meta_features_dict_train, *updated_meta_features_dict_test)

## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type

        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)

        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)


# load r/place dataset
def place_data(text_field, label_field, sr_name_field, meta_data_field, config_dict, fold_number, **kargs):
    train_data, dev_data, test_data = torch_loader.TorchLoader.splits(text_field=text_field, label_field=label_field,
                                                                      sr_name_field=sr_name_field,
                                                                      meta_data_field=meta_data_field,
                                                                      config_dict=config_dict,
                                                                      fold_number=fold_number)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    sr_name_field.build_vocab(train_data, dev_data, test_data)
    # only in case we want to use the meta features, we need to run the data prep to these,
    # otherwise, the meta-data field will be filled with None anyway
    if eval(config_dict['meta_data_usage']['use_meta']):
        data_prep_meta_features(dev_data, test_data, train_data=train_data, update_objects=True)
    batch_size = config_dict['class_model']['nn_params']['batch_size']
    train_data, dev_data, test_data = data.Iterator.splits((train_data, dev_data, test_data),
                                                           batch_sizes=(batch_size, batch_size, batch_size),
                                                           sort=True,
                                                           shuffle=False,
                                                           sort_key=lambda x: len(x.text),
                                                           **kargs)
    return train_data, dev_data, test_data


# setting the torch random seed to the given seed in the configuration file
def set_random_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    random.seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True