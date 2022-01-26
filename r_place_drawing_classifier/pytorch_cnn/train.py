# Authors: Abraham Israeli
# Python version: 3.7
# Last update: 26.01.2021

import os
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from r_place_drawing_classifier.pytorch_cnn.utils import submissions_separation
import gc
import datetime
import random


def train(train_data, dev_data, test_data, model, config_dict, fold):
    if config_dict['cuda']:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['class_model']['nn_params']['lr'])

    steps = 0
    best_acc = 0
    last_step = 0
    min_sent_length = max(model.config_dict['kernel_sizes'])
    # building the saving path location according to info in the config file
    save_dir = os.path.join(config_dict['results_dir'][config_dict['machine']], config_dict['model_version'])
    start_time = datetime.datetime.now()
    # setting the model to train mode (pytorch requires this)
    for epoch in range(1, config_dict['class_model']['nn_params']['epochs']+1):
        corrects = 0
        steps = 0
        empty_text_srs = set()
        # looping over each instance in the train data - we do online learning and not batch learning at all
        # note that the iterator of train data converts the text to int "on the fly" - so no text here, only ints
        model.train()
        for example_idx, cur_example in enumerate(train_data):
            feature, target, sr_idx, explanatory_meta_features = cur_example.text, cur_example.label, cur_example.sr_name, cur_example.meta_data
            sr_name = cur_example.dataset.fields['sr_name'].vocab.itos[int(sr_idx[0])]
            if len(feature) == 0:
                empty_text_srs.add(sr_name)
                continue
            # converting the meta-feature to be explicit None and not list of None (since the iterator of the loop
            # looks at the data as batches, it converts all to be list of values, but here it is only a None value)
            if explanatory_meta_features == [None]:
                explanatory_meta_features = None
            # pulling out the name of the SR in order to use it in the near future

            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            # breaking the long sentence we created into the actual submissions it comes from
            feature_separated = submissions_separation(input_tensor=feature,
                                                       separator_int=cur_example.dataset.fields['text'].vocab.stoi['<sent_ends>'],
                                                       padding_int=cur_example.dataset.fields['text'].vocab.stoi['<pad>'])
            del feature
            gc.collect()
            if config_dict['cuda']:
                feature_separated, explanatory_meta_features, target = feature_separated.cuda(), \
                                                                       explanatory_meta_features, target.cuda()
            optimizer.zero_grad()
            logit = model(x=feature_separated, explanatory_meta_features=explanatory_meta_features)
            logit.unsqueeze_(0)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            del feature_separated
            gc.collect()
            steps += 1
            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            if example_idx % 500 == 0 and example_idx > 0:
                dev_acc = evaluation(dev_data, model, config_dict, dataset='dev', return_proba=False)
                test_acc = evaluation(test_data, model, config_dict, dataset='test', return_proba=False)
                duration = (datetime.datetime.now() - start_time).seconds
                print("Finished handling {} objects in {} sec."
                      "Dev accuracy: {:.4f}. Test accuracy: {:.4f}".format(example_idx, duration, dev_acc, test_acc))
        accuracy = 100.0 * corrects / steps
        duration = (datetime.datetime.now() - start_time).seconds
        print("Along training phase, {} srs weren't included (empty input).".format(len(empty_text_srs)))
        print('\rEnd of epoch [{}] - train accuracy: {:.4f}%({}/{}). '
              'Took us up to now {} sec'.format(epoch, accuracy, corrects, steps, duration), flush=True)
        if epoch % config_dict['mode']['test_interval'] == 0:
            model.eval()
            dev_acc = evaluation(dev_data, model, config_dict, dataset='dev', return_proba=False)
            if dev_acc > best_acc:
                best_acc = dev_acc
                last_step = steps
                if eval(config_dict['saving_options']['save_best']):
                    save(model, save_dir, 'best', fold)
            else:
                if eval(config_dict['class_model']['nn_params']['early_stopping']) \
                        and steps - last_step >= config_dict['class_model']['nn_params']['early_stop_steps']:
                    print('early stop by {} steps.'.format(config_dict['class_model']['nn_params']['early_stop_steps']))
        elif epoch % config_dict['saving_options']['save_interval'] == 0:
            save(model, save_dir, 'snapshot', fold)

    # at the end of all epochs, we will evaluate the test set based on the best model we have
    best_model = model
    best_model.load_state_dict(torch.load(os.path.join(save_dir, 'best_fold_' + str(fold) + ".pt")))
    test_proba = evaluation(data_iter=test_data, model=best_model, config_dict=config_dict,
                            dataset='test', return_proba=True)
    return test_proba


def evaluation(data_iter, model, config_dict, dataset='unknown', return_proba=False):
    model.eval()
    corrects, avg_loss = 0, 0
    min_sent_length = max(model.config_dict['kernel_sizes'])
    proba_res = dict()
    size = 0
    empty_text_srs = set()
    for cur_idx, cur_example in enumerate(data_iter):
        feature, target, sr_idx, explanatory_meta_features = cur_example.text, cur_example.label, cur_example.sr_name, cur_example.meta_data
        sr_name = cur_example.dataset.fields['sr_name'].vocab.itos[int(sr_idx[0])]
        if len(feature) == 0:
            empty_text_srs.add(sr_name)
            continue
        # converting the meta-feature to be explicit None and not list of None (since the iterator of the loop
        # looks at the data as batches, it converts all to be list of values, but here it is only a None value)
        if explanatory_meta_features == [None]:
            explanatory_meta_features = None
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if config_dict['cuda']:
            feature, target = feature.cuda(), target.cuda()
        if len(feature.data[0]) < min_sent_length:
            continue
        # breaking the long sentence we created into the actual submissions it comes from
        feature_separated = submissions_separation(input_tensor=feature,
                                                   separator_int=cur_example.dataset.fields['text'].vocab.stoi[
                                                       '<sent_ends>'],
                                                   padding_int=cur_example.dataset.fields['text'].vocab.stoi['<pad>'])

        # the actual learning phase
        logit = model(x=feature_separated, explanatory_meta_features=explanatory_meta_features)
        logit.unsqueeze_(0)
        cur_proba = torch.nn.functional.softmax(logit, dim=1).data[0].tolist()
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()
        proba_res[sr_name] = (cur_proba, target.data[0].tolist())
        size += 1

    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print("Along evaluation phase, {} srs weren't included (empty input)".format(len(empty_text_srs)))
    print('\nEvaluation ({} dataset)- loss: {:.6f}  acc: {:.4f}%({}/{})'.format(dataset, avg_loss, accuracy,
                                                                              corrects, size))
    gc.collect()
    if return_proba:
        return proba_res
    else:
        return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, fold):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_fold_{}.pt'.format(save_prefix, fold)
    torch.save(model.state_dict(), save_path)
