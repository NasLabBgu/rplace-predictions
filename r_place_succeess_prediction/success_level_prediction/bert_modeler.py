from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, BertForSequenceClassification,AutoConfig
)
import numpy as np
from datasets import load_metric, Dataset, load_dataset
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from os.path import join as opj
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import csv
from scipy.special import softmax
from r_place_success_analysis.success_level_prediction.multi_modal_bert import ModelArguments, MultimodalDataTrainingArguments
from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular


class BertModeler(object):
    def __init__(self, model_name, saving_path, batch_size=64, max_epochs=100, multimodal=True):
        self.model_name = model_name
        self.saving_path = saving_path
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.multimodal = multimodal
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.trainer = None

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    @staticmethod
    def compute_metrics(eval_pred):  # p should be of type EvalPrediction
        """
        an inner function for calculating few metrics along building the DL model
        taken from: https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_utils.py
        :param eval_pred: function
            the function which returns the prediction per instance (returns the logits and the gold label)
        :return: dict
            a dictionary with results (f1 and friends)
        """

        # the logits contain a lot of information, the 0 place are the actual logits and the first place is the
        # embeddings (+logits again)
        logits, labels = eval_pred
        binary_predictions = np.argmax(logits[0], axis=-1)
        porba_predictions = softmax(logits[0], axis=-1)[:, 1]
        assert len(binary_predictions) == len(labels)
        f1 = f1_score(labels, binary_predictions, average='weighted', labels=[0, 1])
        f1_pos = f1_score(labels, binary_predictions, pos_label=1, average='macro', labels=[1])
        f1_neg = f1_score(labels, binary_predictions, pos_label=0, average='macro', labels=[0])
        weighted_precision = precision_score(labels, binary_predictions, average='weighted')
        weighted_recall = recall_score(labels, binary_predictions, average='weighted')
        acc = accuracy_score(labels, binary_predictions)
        auc = roc_auc_score(labels, porba_predictions)
        return {
            'f1_pos': f1_pos,
            'f1_neg': f1_neg,
            'weighted_f1': f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'accuracy': acc,
            'auc': auc
        }

    def train(self, train_dataset, val_dataset):
        training_args = TrainingArguments(output_dir=opj(self.saving_path, 'checkpoints'),
                                          per_device_train_batch_size=self.batch_size,
                                          per_device_eval_batch_size=self.batch_size,
                                          num_train_epochs=self.max_epochs,
                                          evaluation_strategy='epoch',
                                          load_best_model_at_end=True,
                                          optim="adamw_torch",
                                          logging_strategy='epoch', save_strategy='epoch',
                                          logging_dir=opj(self.saving_path, 'logs'))
        # multimodal means we also use numerical features (structural ones)
        if self.multimodal:
            column_info_dict = {
                'text_cols': ['text'],
                'num_cols': [c for c in train_dataset.columns if c not in {'text', 'label'}],
                'cat_cols': [],
                'label_col': 'label',
                'label_list': [0, 1]
            }
            model_args = ModelArguments(model_name_or_path=self.model_name)
            data_args = MultimodalDataTrainingArguments(data_path=self.saving_path,
                                                        combine_feat_method='gating_on_cat_and_num_feats_then_sum',
                                                        column_info=column_info_dict,
                                                        task='classification'
                                                        )
            # the package is usually used in a way that the data is loaded from the disk. This is how we'll do it
            # We save the data into a csv file and then load it
            train_dataset.to_csv(opj(self.saving_path, 'train.csv'))
            val_dataset.to_csv(opj(self.saving_path, 'val.csv'))
            val_dataset.to_csv(opj(self.saving_path, 'test.csv'))
            # we assume here that no categorical features exist. Hence, we set the categorical_encode_type to None
            # if categorical features do exist - need to change it accordingly
            train_dataset, val_dataset, _ = load_data_from_folder(
                data_args.data_path,
                data_args.column_info['text_cols'],
                self.tokenizer,
                label_col=data_args.column_info['label_col'],
                label_list=data_args.column_info['label_list'],
                categorical_cols=data_args.column_info['cat_cols'],
                numerical_cols=data_args.column_info['num_cols'],
                sep_text_token_str=self.tokenizer.sep_token,
                categorical_encode_type=None
            )
            # now we can delete the files
            os.remove(opj(self.saving_path, 'train.csv'))
            os.remove(opj(self.saving_path, 'val.csv'))
            os.remove(opj(self.saving_path, 'test.csv'))
            num_labels = len(np.unique(train_dataset.labels))
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
            )
            cat_features_dim = 0 if train_dataset.cat_feats is None else train_dataset.cat_feats.shape[1]
            numerical_features_dim = 0 if train_dataset.numerical_feats is None else train_dataset.numerical_feats.shape[1]
            tabular_config = TabularConfig(num_labels=num_labels,
                                           cat_feat_dim=cat_features_dim,
                                           numerical_feat_dim=numerical_features_dim,
                                           **vars(data_args))
            config.tabular_config = tabular_config
            model = AutoModelWithTabular.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir
            )

            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
            )
            self.trainer.train()
        else:
            # code concept is taken from here: https://huggingface.co/docs/transformers/training
            train_data = Dataset.from_pandas(train_dataset)
            eval_data = Dataset.from_pandas(val_dataset)
            # dataset should be a list. Each item in the list is a dict with 'label' and text keys
            tokenized_train = train_data.map(self.tokenize_function, batched=True)
            tokenized_eval = eval_data.map(self.tokenize_function, batched=True)
            model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            # TODO: add a fixed see (using set_seed function)
            sw = SummaryWriter(log_dir=opj(self.saving_path, 'tensorboard_logs'))
            self.trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train,
                                   eval_dataset=tokenized_eval,
                                   compute_metrics=self.compute_metrics)
            self.trainer.train()

    def predict_and_eval(self, test_dataset):
        softmax_func = torch.nn.Softmax(dim=1)
        # in case of a multumodal - the data loading is more difficult. besides, almost the same as the regular model
        if self.multimodal:
            column_info_dict = {
                'text_cols': ['text'],
                'num_cols': [c for c in test_dataset.columns if c not in {'text', 'label'}],
                'cat_cols': [],
                'label_col': 'label',
                'label_list': [0, 1]
            }
            data_args = MultimodalDataTrainingArguments(data_path=self.saving_path,
                                                        combine_feat_method='gating_on_cat_and_num_feats_then_sum',
                                                        column_info=column_info_dict,
                                                        task='classification'
                                                        )
            test_dataset.to_csv(opj(self.saving_path, 'test.csv'))
            test_dataset.to_csv(opj(self.saving_path, 'train.csv'))
            test_dataset.to_csv(opj(self.saving_path, 'val.csv'))
            _, _, test_dataset = load_data_from_folder(
                self.saving_path,
                data_args.column_info['text_cols'],
                self.tokenizer,
                label_col=data_args.column_info['label_col'],
                label_list=data_args.column_info['label_list'],
                categorical_cols=data_args.column_info['cat_cols'],
                numerical_cols=data_args.column_info['num_cols'],
                sep_text_token_str=self.tokenizer.sep_token,
                categorical_encode_type=None
            )
            # now we can delete the files
            os.remove(opj(self.saving_path, 'train.csv'))
            os.remove(opj(self.saving_path, 'val.csv'))
            os.remove(opj(self.saving_path, 'test.csv'))
            self.trainer._load_best_model()
            predictions_output = self.trainer.predict(test_dataset)
            proba_pred = softmax_func(torch.from_numpy(predictions_output.predictions[0]))
        else:
            test_data = Dataset.from_pandas(test_dataset)
            tokenized_test = test_data.map(self.tokenize_function, batched=True)
            # making sure the trainer holds the best model
            self.trainer._load_best_model()
            predictions_output = self.trainer.predict(tokenized_test)
            proba_pred = softmax_func(torch.from_numpy(predictions_output.predictions))
        test_measures = predictions_output.metrics
        # converting the prediction back to a numpy array
        proba_pred = list(np.array(proba_pred[:, 1]))
        true_labels = list(predictions_output.label_ids)
        return {'proba_pred': proba_pred, 'true_labels': true_labels, 'measures': test_measures}

    @staticmethod
    def save_results_to_csv(results_file, cv_index, start_time, objects_amount, results):
        """
        given inputs regarding a final run results - write these results into a file
        :param results_file: str
            file of the csv where results should be placed
        :param start_time: datetime
            time when the current result run started
        :param objects_amount: int
            amount of objects in the run was based on, usually it is between 1000-2500n
        :param results: dict
            dictionary with all results. Currently it should contain the following keys: 'accuracy', 'precision', 'recall'
        :return: None
            Nothing is returned, only saving to the file is being done
        """

        file_exists = os.path.isfile(results_file)
        rf = open(results_file, 'a', newline='')
        eval_measures = list(results.keys())
        with rf as output_file:
            dict_writer = csv.DictWriter(output_file,
                                         fieldnames=['timestamp', 'cv_index', 'start_time', 'SRs_amount'] + eval_measures)
            # only in case the file doesn't exist, we'll add a header
            if not file_exists:
                dict_writer.writeheader()

            dict_to_write = {'timestamp': datetime.datetime.now(), 'cv_index': cv_index, 'start_time': start_time,
                             'SRs_amount': objects_amount}
            dict_to_write.update(results)
            dict_writer.writerow(dict_to_write)
        rf.close()
