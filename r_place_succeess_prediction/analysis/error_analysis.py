import numpy as np
import pandas as pd
from collections import Counter
import os
from os.path import join as opj

# configurations
results_dir = "/sise/home/isabrah/reddit_canvas/results/success_analysis"
# 1.8 is the best model of the classification setup
# 2.5 is the best model of the regression setup
model_to_analyze = '3.28' #'1.8'
model_type = 'regression' # has to be either 'classification' or 'regression'

if __name__ == "__main__":
    # pulling out the csv with the predictions
    raw_lvl_res = pd.read_csv(opj(results_dir, model_to_analyze, 'raw_level_res.csv'))
    if model_type == 'classification':
        # false negative/positive predictions
        false_neg_cases = raw_lvl_res[raw_lvl_res['true_value'] == 1].sort_values(by='pred').head(10)
        false_positive_cases = raw_lvl_res[raw_lvl_res['true_value'] == 0].sort_values(by='pred', ascending=False).head(10)
        print(f"Here are the false negative cases: {false_neg_cases}")
        print(f"Here are the false positive cases: {false_positive_cases}")
    if model_type == 'regression':
        raw_lvl_res['delta'] = raw_lvl_res['pred'] - raw_lvl_res['true_value']
        raw_lvl_res_sorted = raw_lvl_res.sort_values(by='delta', ascending=False).copy()
        over_predicted_cases = raw_lvl_res_sorted.head(10)
        under_predicted_cases = raw_lvl_res_sorted.tail(10)
        print(f"Here are the over-predicted cases:\n {over_predicted_cases}")
        print(f"Here are the under-predicted cases:\n {under_predicted_cases}")


