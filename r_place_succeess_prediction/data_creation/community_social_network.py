
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
from os.path import join as opj
import statistics


"""
The purpose of this code is to create network txt files for each community
The historical procedure of creating the network txt files was long and not efficient
Here, for each community we load its object which contains all submissions and comments and find all
interactions between different users (interactions = user A commented user B)

Code example of how to run the code:
import pickle
import os
sr_objects_path = "/data/work/data/reddit_place/sr_objects"
sr_name = 'israel'
cur_sr_obj = pickle.load(open(os.path.join(sr_objects_path, 'sr_obj_' + sr_name + '_.p'), "rb"))
submissions_as_list = cur_sr_obj.submissions_as_list
comments_as_list = cur_sr_obj.comments_as_list
csn_obj = CommunitySocialNetwork(name=sr_name, submissions=submissions_as_list, comments=comments_as_list)
csn_obj.extract_social_relations(remove_self_loops=True)
print(len(csn_obj.social_relations))
csn_obj.create_nx_graph()
csn_obj.extract_structural_features()
"""


class CommunitySocialNetwork(object):

    def __init__(self, name, submissions, comments):
        self.name = name
        self.submissions = submissions
        self.comments = comments
        self.social_relations = None
        self.networkx_graph = None
        self.structural_features = defaultdict()

    def extract_social_relations(self, remove_self_loops=True):
        """

        We assume that the structure of each submission is a tuple of length 6. In the 4-5 places (index 3-4) we expect
        to find the ID of the submission and the user_name respectively
        An example of a tuple: (3, '8 down, 15 to go.', nan, '55atxw', 'asdfnask', '2016-10-01 00:01:46')

        We assume that the structure of each comment is a tuple of length 6. In the 4-5 places (index 3-4) we expect
        to find the ID of the comment and the user_name respectively. Note that the ID here is more complex, it is
        a dictionary with 3 values.
        An example of a tuple: (2, '50/50', '', {'id': 'd88zy62', 'link_id': 't3_55auav', 'parent_id': 't3_55auav'}, 'nikersc', '2016-10-01 00:05:44')

        :param remove_self_loops:
        :param normalize_weights:
        :return:
        """
        try:
            submissions_df = pd.DataFrame([cur_sub[3:5] for cur_sub in self.submissions], columns=['id', 'user_name'])
        except TypeError:
            submissions_df = pd.DataFrame([], columns=['id', 'user_name'])
        try:
            comments_df = pd.DataFrame([list(cur_sub[3].values()) + [cur_sub[4]] for cur_sub in self.comments],
                                       columns=['id', 'link_id', 'parent_id', 'user_name'])
        except TypeError:
            comments_df = pd.DataFrame([], columns=['id', 'link_id', 'parent_id', 'user_name'])
        # adding the 't3_' / 't1_' prefix for latter join operators
        submissions_df['id'] = 't3_' + submissions_df['id']
        comments_df['id'] = 't1_' + comments_df['id']

        # doing 2 joins - one over the submissions and one over the comments
        merged_dfs = pd.merge(comments_df, submissions_df, how="left", left_on='parent_id', right_on='id')
        merged_dfs.columns = ['id', 'link_id', 'parent_id', 'user_name', 'response_to_submission_id',
                              'response_to_submission_name']
        merged_dfs = pd.merge(merged_dfs, comments_df, how="left", left_on='parent_id', right_on='id')
        merged_dfs = merged_dfs.drop(columns=['link_id_y', 'parent_id_y'])
        merged_dfs.columns = ['id', 'link_id', 'parent_id', 'user_name', 'response_to_submission_id',
                              'response_to_submission_name', 'response_to_comment_id', 'response_to_comment_name']
        # filtering some of the non-relevant rows ('deleted' users and those with np.nan)
        merged_dfs_filtered = merged_dfs[(merged_dfs['user_name'] != '[deleted]') &
                                         (merged_dfs['response_to_submission_name'] != '[deleted]') &
                                         (merged_dfs['response_to_comment_name'] != '[deleted]')]
        merged_dfs_filtered = merged_dfs_filtered[(~merged_dfs_filtered['response_to_submission_id'].isnull()) |
                                                  (~merged_dfs_filtered['response_to_comment_name'].isnull())]
        # now pulling the actual relations
        user_connections_based_sub = merged_dfs_filtered[~merged_dfs_filtered['response_to_submission_name'].isnull()][
            ['user_name', 'response_to_submission_name']]
        user_connections_based_com = merged_dfs_filtered[~merged_dfs_filtered['response_to_comment_name'].isnull()][
            ['user_name', 'response_to_comment_name']]
        user_relations_based_sub = list(user_connections_based_sub.itertuples(index=False, name=None))
        user_relations_based_com = list(user_connections_based_com.itertuples(index=False, name=None))
        # adding
        user_relations = user_relations_based_sub + user_relations_based_com
        user_relations_agg = dict(Counter(user_relations))
        if remove_self_loops:
            user_relations_agg = {key: value for key, value in user_relations_agg.items() if key[0] != key[1]}
        self.social_relations = user_relations_agg

    def create_nx_graph(self):
        if self.social_relations is None:
            print("The social_relations attribute has not been set yet. You should call the 'extract_social_relations'"
                  " function first and then the current function. We will do it for you now.")
            self.extract_social_relations()
        nx_graph = nx.DiGraph((node_a, node_b, {'weight': w}) for (node_a, node_b), w in self.social_relations.items())
        self.networkx_graph = nx_graph

    def extract_structural_features(self):
        if self.networkx_graph is None:
            self.create_nx_graph()
        # apply the function from Alex
        undirected_graph = nx.Graph(self.networkx_graph)
        self.structural_features['num_of_nodes'] = self.networkx_graph.number_of_nodes()
        self.structural_features['num_of_edges'] = self.networkx_graph.number_of_edges()
        self.structural_features['num_of_triangles'] = sum(nx.triangles(undirected_graph).values()) / 3
        self.structural_features['is_biconnected'] = 1 if nx.is_biconnected(undirected_graph) else 0
        self.structural_features['num_of_nodes_to_cut'] = len(list(nx.articulation_points(undirected_graph)))
        self.structural_features['density'] = nx.density(self.networkx_graph)

        # now using other functions from the code below to get other features
        self.extract_degrees(return_node_level_values=False)
        self.extract_centrality(return_node_level_values=False)
        self.extract_closeness(return_node_level_values=False)
        self.extract_betweeness(return_node_level_values=False)
        self.extract_components()

    # the next 5 functions are taken from Alex code (/mnt/IseHome/kremians/network_analysis) with few modifications
    def extract_degrees(self, return_node_level_values=False):
        in_degrees = {}
        out_degrees = {}
        for node in self.networkx_graph.nbunch_iter():
            in_degrees[node] = self.networkx_graph.in_degree(node)
            out_degrees[node] = self.networkx_graph.out_degree(node)
        in_degree = self.get_agg_stats(list(in_degrees.values()))
        out_degree = self.get_agg_stats(list(out_degrees.values()))
        self.structural_features['in_degree'] = in_degree
        self.structural_features['out_degree'] = out_degree
        if return_node_level_values:
            return {'in_degree': in_degrees, 'out_degrees': out_degrees}
        else:
            return (in_degree, out_degree), {'in_degree': in_degrees, 'out_degrees': out_degrees}

    def extract_centrality(self, return_node_level_values=False):
        try:
            centrality = nx.degree_centrality(self.networkx_graph)
            centrality_values = list(centrality.values())
            central_dict = self.get_agg_stats(centrality_values)
        except Exception:
            centrality_values = list()
            central_dict = {'avg': 0.0, 'median': 0.0, 'stdev': 0, 'max': 0.0, 'min': 0.0}
        self.structural_features['centrality'] = central_dict
        if return_node_level_values:
            return central_dict, centrality_values
        else:
            return central_dict

    def extract_betweeness(self, return_node_level_values=False):
        try:
            betweenness = nx.betweenness_centrality(self.networkx_graph)
            betweenness_values = list(betweenness.values())
            betweenness_dict = self.get_agg_stats(betweenness_values)
        except Exception:
            betweenness_values = list()
            betweenness_dict = {'avg': 0.0, 'median': 0.0, 'stdev': 0, 'max': 0.0, 'min': 0.0}
        self.structural_features['betweenness'] = betweenness_dict
        if return_node_level_values:
            return betweenness_dict, betweenness_values
        else:
            return betweenness_dict

    def extract_closeness(self, return_node_level_values=False):
        try:
            closeness = nx.closeness_centrality(self.networkx_graph)
            closeness_values = list(closeness.values())
            closeness_dict = self.get_agg_stats(closeness_values)
        except Exception:
            closeness_values = list()
            closeness_dict = {'avg': 0.0, 'median': 0.0, 'stdev': 0, 'max': 0.0, 'min': 0.0}
        self.structural_features['closeness'] = closeness_dict
        if return_node_level_values:
            return closeness_dict, closeness_values
        else:
            return closeness_dict

    def extract_components(self):
        undirected_graph = nx.Graph(self.networkx_graph)
        components_dict = defaultdict(lambda: defaultdict(float))
        components = list(nx.connected_components(undirected_graph))
        components_dict['connected_components']['num'] = len(components)
        components_dict['connected_components']['num_>_2'] = len([comp for comp in components if len(comp) > 2])

        strongly_components = list(nx.strongly_connected_components(self.networkx_graph))
        components_dict['strongly_connected_components']['num'] = len(strongly_components)
        components_dict['strongly_connected_components']['num_>_1'] = len(
            [comp for comp in strongly_components if len(comp) > 1])
        try:
            components_dict['connected_components']['max_component'] = max([len(comp) for comp in components])
            components_dict['strongly_connected_components']['max_component'] = max(
                [len(comp) for comp in strongly_components])
        except ValueError:
            components_dict['strongly_connected_components']['max_component'] = 0
            components_dict['connected_components']['max_component'] = 0
        self.structural_features.update(components_dict)
        return components_dict

    def save_edgelist_file(self, path, f_name):
        if self.networkx_graph is None:
            print("The networkx_graph attribure has not been set yet. You should call the 'create_nx_graph'"
                  " function first and then the current function.")
        nx.write_edgelist(self.networkx_graph, opj(path, f_name), delimiter=',')

    @staticmethod
    def get_agg_stats(values):
        # in case the list is empty
        if not len(values):
            return {'avg': None, 'median': None, 'stdev': None, 'max': None, 'min': None}
        return_dict = dict()
        return_dict['avg'] = statistics.mean(values)
        return_dict['median'] = statistics.median(values)
        return_dict['stdev'] = statistics.stdev(values) if len(values) > 1 else 0
        return_dict['max'] = max(values)
        return_dict['min'] = min(values)
        return return_dict


if __name__ == "__main__":
    import pickle
    import os

    sr_objects_path = "/data/work/data/reddit_place/sr_objects"
    sr_name = 'israel'
    cur_sr_obj = pickle.load(open(os.path.join(sr_objects_path, 'sr_obj_' + sr_name + '_.p'), "rb"))
    submissions_as_list = cur_sr_obj.submissions_as_list
    comments_as_list = cur_sr_obj.comments_as_list
    csn_obj = CommunitySocialNetwork(name=sr_name, submissions=submissions_as_list, comments=comments_as_list)
    csn_obj.extract_social_relations(remove_self_loops=True)
    print(len(csn_obj.social_relations))
    csn_obj.create_nx_graph()
    csn_obj.extract_structural_features()

