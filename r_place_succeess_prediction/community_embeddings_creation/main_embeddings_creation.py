from os.path import join as opj
import pickle
import pandas as pd
import sys
from gensim.models import Doc2Vec
import multiprocessing
import datetime
import os
if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from srs_word_embeddings.embedding_models_creation.posts_yielder import PostsYielder
from r_place_success_analysis.community_embeddings_creation.graph_embeddings import GraphEmbeddings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

extract_doc2vec = False
extract_com2vec = False
extract_snap_embeddings = False
# DO NOT set the next 2 as true - only one of them!
create_doc2vec_before_exper = False
create_doc2vec_while_exper = False
# DO NOT set the next 2 as true - only one of them!
create_graph2vec_before_exper = False
create_graph2vec_while_exper = True

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    # loading the community names of those that participated in r/place
    data_path = "/data/work/data/reddit_place/canvas_annotation_effort_data"
    com_level_gold_label_df = \
        pickle.load(open(opj(data_path, 'success_analysis_target_df_09_02_2022.p'), "rb"))
    com_names = list(set(com_level_gold_label_df.index))
    if extract_doc2vec:
        model_path = '/data/work/data/reddit_place/embedding/embedding_per_sr/3.00/'
        doc2vec_dict = pickle.load(open(opj(model_path, 'communities_doc2vec_model_5_11_2019_dict.p'), "rb"))
        doc2vec_r_place_com = {key: value for key, value in doc2vec_dict.items() if key in com_names}
    if extract_com2vec:
        model_path = '/data/work/data/reddit_place/com2vec_algorithm'
        com2vec_dict = pickle.load(open(opj(model_path, 'com2vec_2015_to_2019_13_4_2019_dict.p'), "rb"))
        com2vec_r_place_com = {key: value for key, value in com2vec_dict.items() if key in com_names}
    if extract_snap_embeddings:
        model_path = '/data/work/data/reddit_place/embedding/embedding_per_sr/SNAP/'
        snap_embeddings = pd.read_csv(opj(model_path, 'web-redditEmbeddings-subreddits.csv'), header=None)
        snap_embeddings.set_index(0, inplace=True, drop=True)
        found_idx = [idx for idx in snap_embeddings.index if idx in com_names]
        snap_embeddings_r_place_com = snap_embeddings.loc[found_idx].to_dict(orient='index')
    if create_doc2vec_while_exper or create_doc2vec_before_exper:
        attributes_to_consider = ['exper_submissions_as_tokens', 'exper_comments_as_tokens'] if create_doc2vec_while_exper else ['submissions_as_tokens', 'comments_as_tokens']
        # now generating a new model (if required)
        posts_yielder_obj = \
            PostsYielder(data_path=opj(data_path, 'drawing_sr_objects'),
                         attributes_to_consider=attributes_to_consider,
                         objects_limit=None, verbose=True)
        # calling the Gensim Doc2Vec constructor
        doc2vec_model_hyper_params = {"vector_size": 300, "window": 3, "min_count": 3,
                                      "sample": 1e-4, "negative": 5, "dm": 1}
        model = Doc2Vec(**doc2vec_model_hyper_params, workers=int(multiprocessing.cpu_count() * 0.75 + 1))
        # building the vocabulary (first iteration over the files)
        model.build_vocab(posts_yielder_obj)
        # training the model (second iteration over the files)
        model.train(posts_yielder_obj, total_examples=model.corpus_count, epochs=5)
        objects_vectors = {doctag_name: model.docvecs[doctag_name] for doctag_name, _ in model.docvecs.doctags.items()}
        # saving the model + the vector representation as a dict
        doc2vec_saving_path = '/data/work/data/reddit_place/embedding/embedding_per_sr/doc2vec'
        while_or_before_exper = 'while' if create_doc2vec_while_exper else 'before'
        obj_vectors_saving_file = opj(doc2vec_saving_path, "doc2vec_dict_" + while_or_before_exper + "_exper" + ".p")
        model_saving_file = opj(doc2vec_saving_path, "doc2vec_model_" + while_or_before_exper + "_exper" + ".model")
        model.save(model_saving_file)
        pickle.dump(objects_vectors, open(obj_vectors_saving_file, "wb"))
        duration = (datetime.datetime.now() - start_time).seconds
        print(f"doc2vec model has finished and saved in {doc2vec_saving_path}. Total time took us: {duration} seconds.")
    if create_graph2vec_before_exper or create_graph2vec_while_exper:
        sr_objects_path = opj(data_path, 'drawing_sr_objects')
        # encoder_hidden_dim must be 256 :|
        ge_obj = GraphEmbeddings(path_to_srs_obj=sr_objects_path, batch_size=20, encoder_hidden_dim=256,
                                 encoder_n_layers=2, epochs=5, create_from_scratch=True)
        while_or_before_exper = 'while' if create_graph2vec_while_exper else 'before'
        ge_obj.create_graph_objects(txt_files_saving_path=opj(os.path.dirname(sr_objects_path), 'txt_net_files'),
                                    before_or_while_exper=while_or_before_exper, graphs_amount=None)
        # do what is required...
        ge_obj.train_embeddings_model(verbose=True)
        # deleting the txt network files folder we created
        #ge_obj.delete_folder(files_path=opj(os.path.dirname(sr_objects_path), 'txt_net_files'))
        # saving the results
        graph2vec_saving_path = '/data/work/data/reddit_place/embedding/embedding_per_sr/graph2vec'
        saving_file = opj(graph2vec_saving_path, "graph2vec_dict_" + while_or_before_exper + "_exper" + ".p")
        pickle.dump(ge_obj.graph_embeddings_dict, open(saving_file, "wb"))
        duration = (datetime.datetime.now() - start_time).seconds
        print(f"graph2vec model finished and saved in {graph2vec_saving_path}. Total time took us: {duration} seconds.")

