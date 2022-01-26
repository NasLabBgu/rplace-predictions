import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from r_place_drawing_classifier.pytorch_cnn.utils import build_embedding_matrix
from collections import defaultdict
#from allennlp.modules.elmo import Elmo, batch_to_ids
import datetime
from allennlp.commands.elmo import ElmoEmbedder
import gc


class CNN_Text(nn.Module):

    def __init__(self, config_dict, text_field, embedding_file, eval_measures):
        super(CNN_Text, self).__init__()
        self.config_dict = config_dict

        V = config_dict['embed_num']
        D = config_dict['embedding']['emb_size']
        C = config_dict['class_num']
        Ci = 1
        Co = config_dict['class_model']['cnn_max_pooling_parmas']['kernel_num']
        Ks = config_dict['kernel_sizes']
        H = config_dict['class_model']['cnn_max_pooling_parmas']['last_mlp_dim']

        # Handling embedding component - either we use Elmo model / glove like pre-trained model / none of these
        # glove/w2vec option
        if eval(config_dict['embedding']['use_pretrained']) and config_dict['embedding']['model_type'] != 'elmo':
            self.embed = nn.Embedding(V, D)
            pre_trained_embedding = build_embedding_matrix(embedding_file, text_field,
                                                           emb_size=config_dict['embedding']['emb_size'])
            self.embed.weight.data.copy_(torch.from_numpy(pre_trained_embedding))
        # elmo option
        elif eval(config_dict['embedding']['use_pretrained']) and config_dict['embedding']['model_type'] == 'elmo':
            options_file = config_dict['embedding']['elmo_options_file']
            weight_file = config_dict['embedding']['elmo_weight_file']
            self.embed = ElmoEmbedder(options_file, weight_file)
            #self.embed.training = False
            #for p in self.embed.parameters():
            #    p.requires_grad = False

        # none of these (just random constant values)
        else:
            self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(config_dict['class_model']['nn_params']['dropout'])
        # case the model should end up using meta features data
        #if eval(config_dict['meta_data_usage']['use_meta']):
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(len(Ks) * Co + config_dict['meta_features_dim'], H),
            torch.nn.Linear(H, int(H/2)),
            #torch.nn.ReLU(),
            torch.nn.Linear(int(H/2), C),
        )
        self.eval_measures = eval_measures
        self.eval_results = defaultdict(list)
        self.text_field = text_field

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, explanatory_meta_features):
        # case the length of the sent is too short (compared to the kernel defined), we skip it
        min_sent_length = max([ker.kernel_size[0] for ker in self.convs1])
        if any([1 if len(cur_sent) < min_sent_length else 0 for cur_sent in x]):
            raise IOError("Length of sentence is too short compared to kernel sizes given! Please fix")

        if eval(self.config_dict['embedding']['use_pretrained']) and self.config_dict['embedding']['model_type'] == 'elmo':
            self.embed.training = False
            x_as_text = [[self.text_field.vocab.itos[cur_idx] for cur_idx in cur_sent] for cur_sent in x]
            embeddings = list(self.embed.embed_sentences(x_as_text))
            x_embedded = torch.Tensor([e[2] for e in embeddings])
        else:
            #start_time = datetime.datetime.now()
            x_embedded = self.embed(x)  # (N, W, D)
            #duration = (datetime.datetime.now() - start_time).seconds
            #print("embed model loading time: {} sec".format(duration))
        x_embedded_unsqueezed = x_embedded.unsqueeze(1)  # (N, Ci, W, D)
        # now converting x to list of 3 Tensors (one for each convolution)
        x_convultioned = [F.relu(conv(x_embedded_unsqueezed)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x_max_pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_convultioned]  # [(N, Co), ...]*len(Ks)
        # concatenate all kernel-sizes (by default there are 3 such ones)
        x_concat = torch.cat(x_max_pooled, 1)
        # calculating the average per column over all instances
        x_concat_avg = torch.mean(input=x_concat, dim=0)
        # adding the explanatory meta features - sorting them by name and then concatinating it to the NN output
        if explanatory_meta_features is not None:
            meta_features_sorted = [value for (key, value) in sorted(explanatory_meta_features[0].items())]
            # case we run on the GPU, we'll convert the data to the GPU
            if self.config_dict['cuda']:
                meta_features_sorted = torch.FloatTensor(meta_features_sorted).cuda()
            x_concat_avg_with_meta = torch.cat([x_concat_avg, torch.tensor(meta_features_sorted)])
            x_dropped_out = self.dropout(x_concat_avg_with_meta)  # (N, len(Ks)*Co)
        else:
            x_dropped_out = self.dropout(x_concat_avg)  # (N, len(Ks)*Co)
        # x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        # x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        # x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        # x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        # this step doesn't change the shape of x_concat, only adds a probability not to take some weights into account
        logit = self.fc1(x_dropped_out)  # (N, C)
        gc.collect()
        return logit

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