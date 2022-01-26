import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_modeling_df(explanatory_features, bert_embedding, normalize_explanatory=True,
                      merge_method='inner', fill_missing=True, value_for_missing=0):
    explanatory_features_df = pd.DataFrame.from_dict(explanatory_features, orient='index')
    if normalize_explanatory and explanatory_features_df.shape[1] > 0:
        normalize_obj = StandardScaler()
        explanatory_features_df = pd.DataFrame(normalize_obj.fit_transform(explanatory_features_df),
                                               columns=explanatory_features_df.columns,
                                               index=explanatory_features_df.index)
    bert_embedding_df = pd.DataFrame.from_dict(bert_embedding, orient='index')
    # in case the explanatory_features_df is empty - we'll force it to be only column-wise empty
    if explanatory_features_df.shape[0] == 0:
        explanatory_features_df = pd.DataFrame(index=explanatory_features.keys())
    # joining the two df of features together
    data_df = pd.merge(left=explanatory_features_df, right=bert_embedding_df, how=merge_method,
                       left_index=True, right_index=True)
    # filling in the missing data by a single number
    if fill_missing:
        data_df.fillna(value_for_missing, inplace=True)

    return data_df
