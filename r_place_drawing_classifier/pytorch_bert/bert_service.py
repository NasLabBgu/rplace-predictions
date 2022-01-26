# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021

from bert_serving.client import BertClient
from bert_serving.server import BertServer
import time
import pickle
import numpy as np
import spacy
from itertools import chain

nlp = spacy.load('en', disable=['parser', 'ner', 'tagger'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))  # using this to break text into sentences


class BertService(object):
    def __init__(self, server_args, config_dict):
        self.servers_args = server_args
        self.config_dict = config_dict
        self.server = BertServer(self.servers_args)
        self.server.start()
        time.sleep(120)
        self.server.max_seq_len = 1000
        # allowing the server to wake up and be ready
        self.bc = BertClient(timeout=config_dict['bert_config']['timeout_in_milliseconds'],  # 2 minutes time out
                             port=config_dict['bert_config']['bert_server_params']['port'],
                             port_out=config_dict['bert_config']['bert_server_params']['port_out'])
        self.bert_model_dim = len(self.bc.encode(['hello world'])[0])

    def get_sr_representation(self, sr_obj, use_sr_obj_tokens=True):
        request_max_size = self.config_dict['bert_config']['request_max_size']
        # looping over all files found in the directory
        # sorting data according to some logic
        if eval(self.config_dict["submissions_sampling"]["should_sample"]):
            sampling_dict = self.config_dict['submissions_sampling']
            subsample_res = sr_obj.subsample_submissions_data(subsample_logic=sampling_dict['sampling_logic'],
                                                              percentage=sampling_dict['percentage'],
                                                              maximum_submissions=sampling_dict['max_subm'],
                                                              seed=self.config_dict['random_seed'], apply_to_tokens=True)
            if subsample_res < 0:
                return None

        if use_sr_obj_tokens:
            # since we want to call BERT as a service with a flat list of sentences, we need to know how
            # many sentences each submissions holds
            all_sentences, sent_weight = self._sentences_prep(sentences_list=sr_obj.submissions_as_tokens, return_weights=True)
            '''
            subm_sent_amount = [len(sat) for sat in sr_obj.submissions_as_tokens for sent in sat ]
            # now we decide the weight of each sentence, according to the amount of senteces "his submission" hols
            # this is useful in order to normalize the weight of each submission (so submissions with lots of sentences
            # will not get too much weight
            sent_weight = [[1/ssa] * ssa for ssa in subm_sent_amount if ssa > 0]
            sent_weight = np.array([item for sublist in sent_weight for item in sublist if item!=[]])
            all_sentences = [sent for submission in sr_obj.submissions_as_tokens for sent in submission if sent != []]
            '''
            # calling the bert module along with breaking all sentences into chunks
            all_sentences_embed = []
            loop_chunks = (len(all_sentences) - 1) // request_max_size + 1
            for i in range(loop_chunks):
                cur_subm_embd = self.bc.encode(all_sentences[i * request_max_size:(i + 1) * request_max_size], is_tokenized=True)
                all_sentences_embed.append(cur_subm_embd)
            # concatenating all the embeddings into one big array
            full_subm_embd = np.vstack(tuple(all_sentences_embed))
            # multiple the matrix by the weight (one for each sentence)
            full_subm_embd = np.multiply(full_subm_embd, sent_weight[:, np.newaxis])
            # averaging over the columns since we want to have fixed representation per SR (and not per sentence)
            return np.mean(full_subm_embd, axis=0)

        else:
            cur_sr_subm_embd_avg = np.zeros(self.bert_model_dim)
            valid_subm_counter = 0
            # looping over each submission in the SR
            for subm_idx, cur_subm in enumerate(sr_obj.submissions_as_list):
                header_is_valid = type(cur_subm[1])is str
                selftext_is_valid = type(cur_subm[2])is str
                # case both header and selftext are not valid - skipping the row
                if not header_is_valid and not selftext_is_valid:
                    continue
                got_response = False
                num_tries = 0
                # looping till we get an answer
                while not got_response and num_tries < 10:
                    try:
                        # both header and selftext are valid
                        if header_is_valid and selftext_is_valid:
                            cur_header = nlp(cur_subm[1])
                            cur_selftext = nlp(cur_subm[2])
                            cur_subm_embd = self.bc.encode([sen.text for sen in chain(cur_header.sents, cur_selftext.sents)
                                                       if sen.text and not sen.text.isspace()]).mean(axis=0)
                            got_response=True
                        # only header is valid
                        elif header_is_valid:
                            cur_header = nlp(cur_subm[1])
                            cur_subm_embd = self.bc.encode([sen.text for sen in cur_header.sents
                                                       if sen.text and not sen.text.isspace()]).mean(axis=0)
                            got_response=True
                        # only selftext is valid
                        elif selftext_is_valid:
                            cur_selftext = nlp(cur_subm[2])
                            cur_subm_embd = self.bc.encode([sen.text for sen in cur_selftext.sents
                                                       if sen.text and not sen.text.isspace()]).mean(axis=0)
                            got_response = True
                    except TimeoutError:
                        # restarting the server, only in case is is not responsive
                        if not self.server.is_ready.is_set() or not self.server.is_alive():
                            server = BertServer(self.args)
                            server.start()
                            time.sleep(180)  # allowing the server to wake up and be ready
                        else:
                            self.bc.timeout = int(self.bc.timeout * 1.5)
                # case we have tried 3 times and still no response from the server, we give up and print the error
                if not got_response and num_tries >= 1:
                    print("While handling sr {} with submission index {}, "
                          "the BERT server failed to provide a response".format(sr_obj.name, subm_idx))

                cur_sr_subm_embd_avg += cur_subm_embd
                valid_subm_counter += 1
                self.bc.timeout = 120000
                print("Passed through {} submission".format(subm_idx))
            # since we need the avg, we will divide the vector by the amount of elements we have added,
            # and then will return this information
            cur_sr_subm_embd_avg /= valid_subm_counter
            return cur_sr_subm_embd_avg

    def close(self):
        self.server.close()

    @staticmethod
    def _sentences_prep(sentences_list, return_weights=True):
        sent_amount_in_each_batch = []
        all_sentences = []
        for current_batch in sentences_list:
            valid_sentences = 0
            for sentence in current_batch:
                if sentence:
                    valid_sentences += 1
                    all_sentences.append(sentence)
            if valid_sentences:
                sent_amount_in_each_batch.append(valid_sentences)
        # now we decide the weight of each sentence, according to the amount of senteces "his submission" hols
        # this is useful in order to normalize the weight of each submission (so submissions with lots of sentences
        # will not get too much weight
        sent_weight = [[1/ssb] * ssb for ssb in sent_amount_in_each_batch]
        sent_weight = np.array([item for sublist in sent_weight for item in sublist])

        if return_weights:
            return all_sentences, sent_weight
        else:
            return all_sentences



