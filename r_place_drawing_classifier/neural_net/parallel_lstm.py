# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021

from .nn_classifier import NNClassifier
import dynet as dy
import numpy as np
import time


class ParallelLstm(NNClassifier):
    """
    Parallel long-short term memory based model, using dynet package
    this class inherits from NNClassifier, and is a special case of a NN model. Model implemented here is a sequence of
    LSTMs models used before an MLP layer is applied. Each sentence in a sr goes through this sequence of LSTMs and only
    the last LSTM model output is taken into account for future usage. In order to aggregate all sentences output,
    we take the average of each node in the hidden layer across all LSTMs.

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
        super(ParallelLstm, self).__init__(model=None, tokenizer=tokenizer, eval_measures=eval_measures, emb_size=emb_size,
                                           hid_size=hid_size, early_stopping=early_stopping, epochs=epochs,
                                           use_meta_features=use_meta_features, seed=seed)
        self.use_bilstm = use_bilstm

    @staticmethod
    def _calc_scores_two_layers(sentences, W_emb, first_lstm, W_mlp, b_mlp, V_mlp, a_mlp, meta_data=None):
        """
        calculating the score for parallel LSTM network (in a specific state along learning phase)
        :param sentences: list
            list of lists of sentences (represented already as numbers and not letters)
        :param first_lstm:

        :param W_mlp: model parameter (dynet obj). size: (hid_size, emb_size + meta_data_dim)
            matrix holding weights of the mlp phase
        :param b_mlp: model parameter (dynet obj). size: (hid_size,)
            vector holding weights of intercept for each hidden state
        :param V_mlp: model parameter (dynet obj). size: (2, hid_size)
            matrix holding weights of the logisitc regression phase. 2 is there due to the fact we are in a binary
            classification
        :param a_mlp: model parameter (dynet obj). size: (1,)
            intercept value for the logistic regression phase
        :param meta_data: dict or None
            meta data features for the model. If None - meta data is not used
        :return: dynet parameter. size: (2,)
            prediction of the instance to be a drawing one according to the model (vector of 2, first place is the
            probability to be a drawing team)
        """
        dy.renew_cg()
        word_embs = [[dy.lookup(W_emb, w) for w in words] for words in sentences]
        first_init = first_lstm.initial_state()
        first_embs=[]
        for wb in word_embs:
            first_embs.append(first_init.transduce(wb))
        last_comp_in_first_layer = [i[-1] for i in first_embs]
        # calculating the avg over all last components of the LSTMs
        # if wanted to take the maximum, one can use dy.emax instead of dy.average (but it is not too recommended)
        first_layer_avg = dy.average(last_comp_in_first_layer)
        if meta_data is None:
            h = dy.tanh((W_mlp * first_layer_avg) + b_mlp)
            prediction = dy.logistic((V_mlp * h) + a_mlp)
        else:
            meta_data_ordered = [value for key, value in sorted(meta_data.items())]
            meta_data_vector = dy.inputVector(meta_data_ordered)
            first_layer_avg_and_meta_data = dy.concatenate([first_layer_avg, meta_data_vector])
            h = dy.tanh((W_mlp * first_layer_avg_and_meta_data) + b_mlp)
            prediction = dy.logistic((V_mlp * h) + a_mlp)
        return prediction

    def fit_predict(self, train_data, test_data, embedding_file=None):
        """
        fits a parallel LSTM model
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
        # case we wish to use meta features along modeling, we need to prepare the SRs objects for this
        if self.use_meta_features:
            train_meta_data, test_meta_data = \
                self.data_prep_meta_features(train_data=train_data, test_data=test_data, update_objects=False)
            meta_data_dim = len(train_meta_data[list(train_meta_data.keys())[0]])
        else:
            train_meta_data = None
            test_meta_data = None
            meta_data_dim = 0
        # next we are creating the input for the algorithm. train_data_for_dynet will contain list of lists. Each inner
        # list contains the words index relevant to the specific sentence
        train_data_for_dynet = list(self.get_reddit_sentences(sr_objects=train_data))
        train_data_names = [i[2] for i in train_data_for_dynet]  # list of train sr names
        train_data_for_dynet = [(i[0], i[1]) for i in train_data_for_dynet]

        # test_data_for_dynet will contain list of lists. Each inner list contains the words index relevant to the
        # specific sentence
        test_data_for_dynet = list(self.get_reddit_sentences(sr_objects=test_data))
        test_data_names = [i[2] for i in test_data_for_dynet]   # list of test sr names
        # Need to check here that the order is saved!!!!
        test_data_for_dynet = [(i[0], i[1]) for i in test_data_for_dynet]

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
        first_lstm = dy.LSTMBuilder(1, self.emb_size, self.hid_size, model)    # Forward LSTM
        # Last layer with network is an MLP one, case we are not using meta data, meta_data_dim will be zero
        # and hence not relevant
        W_mlp = model.add_parameters((self.hid_size, self.hid_size + meta_data_dim))
        b_mlp = model.add_parameters(self.hid_size)
        V_mlp = model.add_parameters((self.ntags, self.hid_size))
        a_mlp = model.add_parameters(1)

        mloss = [0.0, 0.0]  # we always save the current run loss and the prev one (for early stopping purposes
        for ITER in range(self.epochs):
            # checking the early stopping criterion
            if self.early_stopping and (ITER >= (self.epochs * 1.0 / 2)) \
                    and ((mloss[0]-mloss[1]) * 1.0 / mloss[0]) <= 0.01:
                print("Early stopping has been applied since improvement was not greater than 1%")
                break
            # Perform training
            start = time.time()
            cur_mloss = 0.0
            for idx, (sentences, tag) in enumerate(train_data_for_dynet):
                cur_meta_data = train_meta_data[train_data_names[idx]] if self.use_meta_features else None
                my_loss =\
                    dy.pickneglogsoftmax(self._calc_scores_two_layers(sentences=sentences, W_emb=W_emb,
                                                                      first_lstm=first_lstm, W_mlp=W_mlp, b_mlp=b_mlp,
                                                                      V_mlp=V_mlp, a_mlp=a_mlp,
                                                                      meta_data=cur_meta_data), tag)
                cur_mloss += my_loss.value()
                my_loss.backward()
                trainer.update()
            # updating the mloss for early stopping purposes
            mloss[0] = mloss[1]
            mloss[1] = cur_mloss
            print("iter %r: train loss/sr=%.4f, time=%.2fs" % (ITER, cur_mloss / len(train_data_for_dynet),
                                                               time.time() - start))
            # Perform testing validation (at the end of current epoch)
            test_correct = 0.0
            for idx, (words, tag) in enumerate(test_data_for_dynet):
                cur_meta_data = test_meta_data[test_data_names[idx]] if self.use_meta_features else None
                scores = self._calc_scores_two_layers(sentences=words, W_emb=W_emb, first_lstm=first_lstm,
                                                      W_mlp=W_mlp, b_mlp=b_mlp, V_mlp=V_mlp, a_mlp=a_mlp,
                                                      meta_data=cur_meta_data).npvalue()
                predict = np.argmax(scores)
                if predict == tag:
                    test_correct += 1
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(test_data_for_dynet)))
        # Perform testing validation after all ephocs ended
        test_correct = 0.0
        test_predicitons = []
        for idx, (words, tag) in enumerate(test_data_for_dynet):
            cur_meta_data = test_meta_data[test_data_names[idx]] if self.use_meta_features else None
            cur_score = self._calc_scores_two_layers(sentences=words, W_emb=W_emb, first_lstm=first_lstm,
                                                     W_mlp=W_mlp, b_mlp=b_mlp, V_mlp=V_mlp, a_mlp=a_mlp,
                                                     meta_data=cur_meta_data).npvalue()
            # adding the prediction of the sr to draw (to be label 1) and calculating the acc on the fly
            test_predicitons.append(cur_score[1])
            predict = np.argmax(cur_score)
            if predict == tag:
                test_correct += 1

        y_test = [a[1] for a in test_data_for_dynet]
        self.calc_eval_measures(y_true=y_test, y_pred=test_predicitons, nomalize_y=True)
        print("final test acc=%.4f" % (test_correct / len(y_test)))

        return self.eval_results, model, test_predicitons
