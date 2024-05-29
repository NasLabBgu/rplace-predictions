import os
from os.path import join as opj
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
import re
import pandas as pd
import multiprocessing as mp
import datetime
import shutil
from r_place_success_analysis.data_creation.community_social_network import CommunitySocialNetwork
from dig.sslgraph.dataset import get_dataset
from dig.sslgraph.utils import Encoder
from dig.sslgraph.method.contrastive.model import InfoGraph
from dig.sslgraph.evaluation.eval_graph import GraphUnsupervised
from torch_geometric.loader import DataLoader

torch.manual_seed(1984)
"""
The purpose of this code is to create graph embeddings for each community
We first create a txt edge file for each community, then we run a learning process using  torch_geometric

Code example:
sr_objects_path = "/data/work/data/reddit_place/canvas_annotation_effort_data/drawing_sr_objects"
# encoder_hidden_dim must be 256 :|
ge_obj = GraphEmbeddings(path_to_srs_obj=sr_objects_path, batch_size=5, encoder_hidden_dim=256,
                         encoder_n_layers=2, epochs=3, create_from_scratch=True)
ge_obj.create_graph_objects(txt_files_saving_path=opj(os.path.dirname(sr_objects_path), 'txt_net_files'),
                            graphs_amount=None)
# do what is required...
ge_obj.train_embeddings_model(verbose=True)
# deleting both the txt network files folder we created and the processed folder, created by the DIG package
ge_obj.delete_folder(files_path=opj(os.path.dirname(sr_objects_path), 'txt_net_files'))
print(len(ge_obj.graph_embeddings_dict))
"""


class GraphEmbeddings(object):
    def __init__(self, path_to_srs_obj, batch_size=100, encoder_hidden_dim=256, encoder_n_layers=2, epochs=5,
                 create_from_scratch=True):
        self.path_to_srs_obj = path_to_srs_obj
        self.data_loader = None

        self.epochs = epochs
        self.batch_size = batch_size
        # currently only works with 256 :|
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_n_layers = encoder_n_layers
        self.create_from_scratch = create_from_scratch
        self.graph_embeddings_dict = None

    def load_and_save_social_net(self, idx, sr_obj_path, saving_path, before_or_while_exper='before'):
        cur_sr_obj = pickle.load(open(opj(self.path_to_srs_obj, sr_obj_path), "rb"))
        cur_sr_name = cur_sr_obj.name
        if before_or_while_exper == 'before':
            csn_obj = CommunitySocialNetwork(name=cur_sr_name, submissions=cur_sr_obj.submissions_as_list,
                                             comments=cur_sr_obj.comments_as_list)
        elif before_or_while_exper == 'while':
            csn_obj = CommunitySocialNetwork(name=cur_sr_name, submissions=cur_sr_obj.exper_submissions_as_list,
                                             comments=cur_sr_obj.exper_comments_as_list)
        else:
            raise IOError("Not a valid 'before_or_while_exper' string value, most be either 'before' or 'while'")
        csn_obj.extract_social_relations(remove_self_loops=True)
        if csn_obj.social_relations is None or len(csn_obj.social_relations) == 0:
            return -1
        csn_obj.create_nx_graph()
        csn_obj.save_edgelist_file(path=saving_path, f_name=cur_sr_name+'.txt')
        return 0

    @staticmethod
    def delete_folder(files_path):
        try:
            shutil.rmtree(files_path)
        except (NotADirectoryError, FileNotFoundError) as e:
            os.remove(files_path)
        return 0

    def create_graph_objects(self, txt_files_saving_path, before_or_while_exper='before', graphs_amount=None):
        start_time = datetime.datetime.now()
        # creating for each sr object a txt file of the social graph - doing it in a multiprocess way
        srs_obj_f_names = sorted([f for f in os.listdir(self.path_to_srs_obj) if re.match(r'.*sr_obj_.*\.p', f)])
        # this specific one causes problems - I don't know why :|
        srs_obj_f_names = [sofn for sofn in srs_obj_f_names if sofn != 'drawing_sr_obj_airforce.p']
        # creating a folder where the txt will be saved (we can later delete this folder
        if not os.path.exists(txt_files_saving_path):
            os.makedirs(txt_files_saving_path)
        data_for_poll = [(idx, opj(self.path_to_srs_obj, sofn), txt_files_saving_path, before_or_while_exper)
                         for idx, sofn in enumerate(srs_obj_f_names)]
        if graphs_amount is not None:
            # Index 25 causes problems. Besides that, up to index 700 all is working.
            data_for_poll = data_for_poll[0:graphs_amount]
        duration = (datetime.datetime.now() - start_time).seconds
        print(f"Starting the pool process of ({len(data_for_poll)} instances). Time up to now: {duration} seconds.")
        pool = mp.Pool(processes=int(mp.cpu_count() * 0.75 + 1))
        #pool = mp.Pool(processes=1)
        with pool as pool:
            results = pool.starmap(self.load_and_save_social_net, data_for_poll)
        duration = (datetime.datetime.now() - start_time).seconds
        print(f"Creation of txt files for each social network ended in {duration} seconds. For {abs(sum(results))} "
              f"communities no file has been created as no users interactions were found.")
        dataset = MyOwnDataset(root=os.path.dirname(self.path_to_srs_obj), success_level_dict=None)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"Creation of a graph dataset has ended in {duration} seconds. Number of graphs: {len(dataset)}")
        self.data_loader = data_loader
        return data_loader

    def train_embeddings_model(self, verbose=True):
        start_time = datetime.datetime.now()
        # if we want to create embeddings from scratch, we need to delete historical files (located under 'processed')
        if self.create_from_scratch:
            self.delete_folder(files_path=opj(os.path.dirname(self.path_to_srs_obj), 'processed'))
        x_dim = self.data_loader.dataset[0].x.shape[1]
        encoder = Encoder(feat_dim=x_dim, hidden_dim=self.encoder_hidden_dim, n_layers=self.encoder_n_layers, gnn='gin', node_level=True)
        infograph = InfoGraph(self.encoder_hidden_dim * self.encoder_n_layers, n_dim=self.encoder_hidden_dim, graph_level=True)
        adam_optim = torch.optim.Adam(params=encoder.parameters(), lr=0.001, weight_decay=0)
        pretrain_loader = DataLoader(self.data_loader.dataset, self.batch_size, shuffle=True) # NEED TO SET HERE A SEED!!!
        # this is the actual training part
        for idx, enc in enumerate(infograph.train(encoder, pretrain_loader, adam_optim, self.epochs, True)):
            if verbose and idx % 1 == 0 and idx > 0:
                duration = (datetime.datetime.now() - start_time).seconds
                print(f"\nWe are inside the 'train_embeddings_model' function. Passes over {idx} epochs so far."
                      f"Duration up to now: {duration} seconds")
        # after training, we apply the encoder to get the actual embeddings per graph
        embeddings_per_sr = dict()
        for data in pretrain_loader:
            embed = encoder(data)
            cur_com_representations = {n: list(e.cpu().detach().numpy()) for n, e in zip(data.name, embed[0])}
            embeddings_per_sr.update(cur_com_representations)
        self.graph_embeddings_dict = embeddings_per_sr
        return embeddings_per_sr


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, success_level_dict=None, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.success_level_dict = success_level_dict

    @property
    def raw_file_names(self):
      txt_files = os.listdir(opj(self.root, 'txt_net_files'))
      return [opj(self.root, 'txt_net_files', cur_f) for cur_f in txt_files]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
      pass
      # Download to `self.raw_dir`.

    def generate_graph_from_txt_file(self, txt_file_path, label=None, name=None):
        net_df = pd.read_csv(txt_file_path, sep=',', header=None, names=['node_A', 'node_B', 'interaction'])
        net_df['interaction'] = net_df['interaction'].apply(lambda i: eval(i)['weight'])
        net_names = set(list(net_df['node_A']) + list(net_df['node_B']))
        net_names_dict = {name: idx for idx, name in enumerate(net_names)}
        net_edges = [(net_names_dict[cur_row['node_A']], net_names_dict[cur_row['node_B']], cur_row['interaction']) for
                     idx, cur_row in net_df.iterrows()]
        edge_index = torch.tensor([[cur_edge[0] for cur_edge in net_edges],
                                   [cur_edge[1] for cur_edge in net_edges]], dtype=torch.long)
        # net_names_sorted_by_key = sorted(net_names_dict, key=net_names_dict.get)
        # users_info = pickle.load(open("users_info_14_3_2021.p", "rb"))
        x_matrix = self.generate_x_matrix(net_df, users_info_dict=None)
        edge_weight = torch.tensor([cur_edge[2] for cur_edge in net_edges])
        if label is not None:
            graph_data_obj = Data(x=x_matrix, edge_index=edge_index, edge_attr=edge_weight,
                                  y=torch.LongTensor([label]), name=name)
        else:
            graph_data_obj = Data(x=x_matrix, edge_index=edge_index, edge_attr=edge_weight, name=name)
        return graph_data_obj

    def process(self):
        # Read data into huge `Data` list.
        txt_files = os.listdir(opj(self.root, 'txt_net_files'))
        file_names = [opj(self.root, 'txt_net_files', cur_f) for cur_f in txt_files]
        data_list = list()
        for idx, f in enumerate(file_names):
            if idx % 1 == 0 and idx > 0:
                print(f"We are in the 'process' function, passed over {idx} instances")
            graph_name = f.split('/')[-1].split('.txt')[0]
            data_list.append(self.generate_graph_from_txt_file(txt_file_path=f, label=None, name=graph_name))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def generate_x_matrix(self, net_df, users_info_dict=None):
        net_names = set(list(net_df['node_A']) + list(net_df['node_B']))
        net_names_dict = {name: idx for idx, name in enumerate(net_names)}
        net_edges = [(net_names_dict[cur_row['node_A']], net_names_dict[cur_row['node_B']], cur_row['interaction']) for
                     idx, cur_row in net_df.iterrows()]
        edge_index = torch.tensor([[cur_edge[0] for cur_edge in net_edges],
                                   [cur_edge[1] for cur_edge in net_edges]], dtype=torch.long)
        net_names_sorted_by_key = sorted(net_names_dict, key=net_names_dict.get)
        out_degree = net_df.groupby('node_A').sum()['interaction'].to_dict()
        in_degree = net_df.groupby('node_B').sum()['interaction'].to_dict()
        degree = dict()
        for key, value in out_degree.items():
            degree[key] = value
        for key, value in in_degree.items():
            if key not in degree:
                degree[key] = value
            else:
                degree[key] += value
        # this is good in case we have a single X feature (e.g., degree value of each node)
        if users_info_dict is None:
            x_values_as_list = [degree[n] for n in net_names_sorted_by_key]
            x_matrix = torch.transpose(torch.FloatTensor(x_values_as_list).reshape((1, -1)), 0, 1)
            return x_matrix
        # this is good in case we have more than a single feature
        else:
            x_matrix_as_list = list()
            for n in net_names_sorted_by_key:
                try:
                    updated_users_info_dict = self.extend_users_info(u_info_dict=users_info_dict[n])
                    new_row = list(updated_users_info_dict.values())
                    new_row += [degree[n]]
                    x_matrix_as_list.append(new_row)
                except KeyError:
                    print(f"Error while trying to get user {n}")
            return torch.FloatTensor(x_matrix_as_list)

    @staticmethod
    def extend_users_info(u_info_dict):
        if u_info_dict['submissions'] > 0:
            u_info_dict['submission_scores_norm'] = u_info_dict['submission_scores'] / u_info_dict['submissions']
            u_info_dict['comments_received_norm'] = u_info_dict['comments_received'] / u_info_dict['submissions']
            u_info_dict['submissions_unique_srs_norm'] = u_info_dict['submissions_unique_srs'] / u_info_dict[
                'submissions']
        else:
            u_info_dict['submission_scores_norm'] = 0.0
            u_info_dict['comments_received_norm'] = 0.0
            u_info_dict['submissions_unique_srs_norm'] = 0.0

        if u_info_dict['comments'] > 0:
            u_info_dict['comments_scores_norm'] = u_info_dict['comments_scores'] / u_info_dict['comments']
            u_info_dict['comments_unique_srs_norm'] = u_info_dict['comments_unique_srs'] / u_info_dict['comments']
        else:
            u_info_dict['comments_scores_norm'] = 0.0
            u_info_dict['comments_unique_srs_norm'] = 0.0

        return u_info_dict

