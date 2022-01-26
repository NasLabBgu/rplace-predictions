# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021

from .nn_classifier import NNClassifier
import random
import dynet as dy
import numpy as np
import time


class SinglelLstm(NNClassifier):
    """
    Single long-short term memory based model, using dynet package
    this class inherits from NNClassifier, and is a special case of a NN model. Model implemented here is a single
    LSTM model before an MLP layer is applied

    Parameters
    ----------
    tokenizer: tokenizer object
        this tokenizer will be used (if needed) to tokenize the text
        example of such definition:
        >>>submission_dp_obj = RedditDataPrep(is_submission_data=True, remove_stop_words=False, most_have_regex=None)
        >>>reddit_tokenizer = submission_dp_obj.tokenize_text
    eval_measures: dict
        dictionary of evlaution measures to use for measuring the model's performance. Each key is a string of
        the measure name, each value is a function to be used for evaluation (e.g. {'accuracy': accuracy_score})
    emb_size: int, default: 100
        size of the embedding vector. Note it mush be aligned with the size of the matrix being used for embedding
        in case we use external source (i.e., if we use glove pre trained embedding matrix, it must be aligned with it)
    hid_size: int, default: 100
        hidden size of the lstm network, this is used along all the architecture
    early_stopping: bool, default: True
        whether to apply early stopping logic along the algorithm
    epochs: int, default: 10
        max number of epochs to apply. Eventually, this number can be lowet due to early_stopping logic
    use_meta_features: bool, default: True
        whether or not to use meta features for modeling
    seed: int, default: 1984
        the random seed to be used along execution
    use_bilstm: boolean, default: False
        whether or not to apply bi directional LSTM model to each sentence (reading the sentence from start to end
        as well as from end to start along modeling)

    Attributes
    ----------
    """

    def __init__(self, tokenizer, eval_measures, emb_size=100, hid_size=100, early_stopping=True,
                 epochs=10, use_meta_features=True, seed=1984, use_bilstm=False):
        super(SinglelLstm, self).__init__(model=None, tokenizer=tokenizer, eval_measures=eval_measures, emb_size=emb_size,
                                          hid_size=hid_size, early_stopping=early_stopping, epochs=epochs,
                                          use_meta_features=use_meta_features, seed=seed)
        self.use_bilstm = use_bilstm

    # A function to calculate scores for one value
    def _calc_scores_single_layer_lstm(self, words, W_emb, fwdLSTM, bwdLSTM, W_sm, b_sm, normalize_results=True):
        """
        calculating the score for an LSTM network (in a specific state along learning phase)
        :param words: list
            list of words representing a sentence (represented already as numbers and not letters)
        :param W_emb: lookup parameter (dynet obj). size: (emb_size x nwords)
            matrix holding the word embedding values
        :param fwdLSTM:

        :param bwdLSTM:

        :param W_sm: model parameter (dynet obj). size: (hid_size, emb_size + meta_data_dim)
            matrix holding weights of the mlp phase
        :param b_sm: model parameter (dynet obj). size: (hid_size,)
            vector holding weights of intercept for each hidden state
        :param normalize_results:

        :return: dynet parameter. size: (2,)
            prediction of the instance to be a drawing one according to the model (vector of 2, first place is the
            probability to be a drawing team)
        """
        dy.renew_cg()
        # embed the words
        word_embs = [dy.lookup(W_emb, x) for x in words]
        fwd_init = fwdLSTM.initial_state()
        fwd_embs = fwd_init.transduce(word_embs)
        # case we wish to use bi directional LSTM model
        if self.use_bilstm:
            bwd_init = bwdLSTM.initial_state()
            bwd_embs = bwd_init.transduce(reversed(word_embs))
            score_not_normalized = W_sm * dy.concatenate([fwd_embs[-1], bwd_embs[-1]]) + b_sm
        else:
            score_not_normalized = W_sm * dy.concatenate([fwd_embs[-1]]) + b_sm

        # case we wish to normalize results (by default we do want to)
        if normalize_results:
            return dy.softmax(score_not_normalized)
        else:
            return score_not_normalized

    def fit_predict(self, train_data, test_data, embedding_file=None):
        """
        fits a single LSTM model
        :param train_data: list
            list of sr objects to be used as train set
        :param test_data: list
            list of sr objects to be used as train set
        :param embedding_file: str
            the path to the exact embedding file to be used. This should be a txt file, each row represents
            a word and it's embedding (separated by whitespace). Example can be taken from 'glove' pre-trained models
            If None, we build an embedding from random normal distribution
        :return: tuple
            tuple with 3 variables:
            self.eval_results, model, test_predicitons
            1. eval_results: dictionary with evaluation measures over the test set
            2. model: the MLP trained model which was used
            3. test_predicitons: list with predictions to each sr in the test dataset
        """
        # Start DyNet and define trainer
        model = dy.Model()
        trainer = dy.AdamTrainer(model)

        # Define the model
        # Word embeddings part
        if embedding_file is None:
            W_emb = model.add_lookup_parameters((self.nwords, self.emb_size))
        else:
            external_embedding = self.build_embedding_matrix(embedding_file)
            W_emb = model.add_lookup_parameters((self.nwords, self.emb_size), init=external_embedding)
        fwdLSTM = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Forward LSTM
        bwdLSTM = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Backward LSTM
        if self.use_bilstm:
            W_sm = model.add_parameters((self.ntags, 2 * self.hid_size))      # Softmax weights
        else:
            W_sm = model.add_parameters((self.ntags, self.hid_size))          # Softmax weights
        b_sm = model.add_parameters(self.ntags)                               # Softmax bias
        for ITER in range(self.epochs):
            # Perform training
            random.seed(self.seed)
            random.shuffle(train_data)
            train_loss = 0.0
            start = time.time()
            for idx, (words, tag) in enumerate(train_data):
                my_loss = dy.pickneglogsoftmax(self._calc_scores_single_layer_lstm(words=words, W_emb=W_emb,
                                                                                   fwdLSTM=fwdLSTM, bwdLSTM=bwdLSTM,
                                                                                   W_sm=W_sm, b_sm=b_sm,
                                                                                   normalize_results=True), tag)
                train_loss += my_loss.value()
                my_loss.backward()
                trainer.update()
            print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train_data), time.time() - start))
            # Perform testing validation
            test_correct = 0.0
            for words, tag in test_data:
                scores = self._calc_scores_single_layer_lstm(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM,
                                                             bwdLSTM=bwdLSTM, W_sm=W_sm, b_sm=b_sm).npvalue()
                predict = np.argmax(scores)
                if predict == tag:
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test_data)))
        # Perform testing validation after all batches ended
        all_scores = []
        for words, tag in test_data:
            cur_score = \
                self._calc_scores_single_layer_lstm(words=words, W_emb=W_emb, fwdLSTM=fwdLSTM, bwdLSTM=bwdLSTM,
                                                    W_sm=W_sm, b_sm=b_sm, normalize_results=True).npvalue()
            all_scores.append(cur_score)
        return all_scores
