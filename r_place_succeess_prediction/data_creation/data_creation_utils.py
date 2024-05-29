import numpy as np
import pandas as pd
from scipy.stats import entropy
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path
from collections import defaultdict, Counter
import gc
import os
from os.path import join as opj
import json
import re
import tldextract
import sys

if sys.platform == 'linux':
    sys.path.append('/data/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_drawing_classifier.utils import get_submissions_subset, get_comments_subset


def load_com_meta_data(data_path, add_handly_labeled_data=True):
    handly_labeled_com = {'monalisaclan': {'creation_epoch': 1491008467, 'subscribers': 680},
                          'starryknights': {'creation_epoch': 1491094867, 'subscribers': 285},
                          'placestart': {'creation_epoch': 1491094867, 'subscribers': 1600},
                          'placesnek': {'creation_epoch': 1491008467, 'subscribers': 129},
                          'grancolombia': {'creation_epoch': 1491094867, 'subscribers': 71},
                          'placetardis': {'creation_epoch': 1491094867, 'subscribers': 33},
                          # 'loliconsunite': {'creation_epoch': , 'subscribers': },
                          'tinytardis': {'creation_epoch': 1491094867, 'subscribers': 78},
                          'heyyea': {'creation_epoch': 1491181267, 'subscribers': 244},
                          'thebutton': {'creation_epoch': 1422925267, 'subscribers': 130000},
                          'anilist': {'creation_epoch': 1404349267, 'subscribers': 293},
                          'goodboye': {'creation_epoch': 1491094867, 'subscribers': 3900},
                          # 'cckufiprfashlewoli0': {'creation_epoch': , 'subscribers': },
                          'theid': {'creation_epoch': 1452042067, 'subscribers': 36},
                          'theblackvoid': {'creation_epoch': 1491008467, 'subscribers': 2800},
                          'federationplace': {'creation_epoch': 1491094867, 'subscribers': 0},
                          'teamcube': {'creation_epoch': 1491008467, 'subscribers': 14},
                          'pinkvomitmonster': {'creation_epoch': 1491008467, 'subscribers': 173},
                          'russiaclan': {'creation_epoch': 1491094867, 'subscribers': 0},
                          'farmcarrots': {'creation_epoch': 1491008467, 'subscribers': 245},
                          'mcgillrobotics': {'creation_epoch': 1491094867, 'subscribers': 16},
                          'manningplace': {'creation_epoch': 1491181267, 'subscribers': 67},
                          'transflagplace': {'creation_epoch': 1491008467, 'subscribers': 360},
                          'armok': {'creation_epoch': 1491008467, 'subscribers': 102},
                          'placeportalpattern': {'creation_epoch': 1491008467, 'subscribers': 92},
                          'chelsea': {'creation_epoch': 1213664467, 'subscribers': 11200},
                          'thehulk': {'creation_epoch': 1491008467, 'subscribers': 45},
                          'hoodr': {'creation_epoch': 1491008467, 'subscribers': 13},
                          'nvade': {'creation_epoch': 1491094867, 'subscribers': 45},
                          'touaoii': {'creation_epoch': 1491181267, 'subscribers': 116},
                          'robigalia': {'creation_epoch': 1491354067, 'subscribers': 1},
                          'afip': {'creation_epoch': 1491008467, 'subscribers': 142},
                          'growthetree': {'creation_epoch': 1491008467, 'subscribers': 74},
                          'rainbowgrid': {'creation_epoch': 1491181267, 'subscribers': 0},
                          # 'centuryclub': {'creation_epoch': , 'subscribers': },
                          'americanflaginplace': {'creation_epoch': 1491008467, 'subscribers': 620},
                          'pgang': {'creation_epoch': 1491181267, 'subscribers': 6},
                          'placehfarts': {'creation_epoch': 1491008467, 'subscribers': 15},
                          # 'britaly': {'creation_epoch': , 'subscribers': },
                          'edditworldcongress': {'creation_epoch': 1491181267, 'subscribers': 112},
                          'cryptofr': {'creation_epoch': 1491008467, 'subscribers': 4100},
                          # 'rathub': {'creation_epoch': , 'subscribers': },
                          'nagisamomoe': {'creation_epoch': 1457744467, 'subscribers': 239},
                          'thepointing': {'creation_epoch': 1491008467, 'subscribers': 8},
                          # 'jakkid166': {'creation_epoch': , 'subscribers': },
                          'epublica': {'creation_epoch': 1491008467, 'subscribers': 38},
                          'edboxes': {'creation_epoch': 1491008467, 'subscribers': 13},
                          # 'place00': {'creation_epoch': , 'subscribers': },
                          'the_r': {'creation_epoch': 1491094867, 'subscribers': 8},
                          'gioj': {'creation_epoch': 1424739667, 'subscribers': 3},
                          # 'protectthet': {'creation_epoch': , 'subscribers': },
                          # 'all': {'creation_epoch': , 'subscribers': },
                          'clanicahn': {'creation_epoch': 1449018067, 'subscribers': 0},
                          'greyblob': {'creation_epoch': 1491008467, 'subscribers': 39},
                          'williamsthing': {'creation_epoch': 1452387667, 'subscribers': 1},
                          'guardiansofdatboi': {'creation_epoch': 1491008467, 'subscribers': 23},
                          'ruinscraft': {'creation_epoch': 1402016467, 'subscribers': 73},
                          'compsoc': {'creation_epoch': 1240016467, 'subscribers': 59}}
    com_meta_data_dict = defaultdict(list)
    with open(opj(data_path, 'srs_meta_data_102016_to_032017.json')) as f:
        for idx, line in enumerate(f):
            cur_line = json.loads(line)
            cur_sr_name = str(cur_line['display_name']).lower()
            # case we already 'met' this sr, we'll take the max out of all subscribers amount we see
            if cur_sr_name in com_meta_data_dict:
                com_meta_data_dict[cur_sr_name] = [max(cur_line['subscribers'], com_meta_data_dict[cur_sr_name][0]),
                                              cur_line['created']]

            elif cur_line['subscribers'] is not None:
                com_meta_data_dict[cur_sr_name] = [cur_line['subscribers'], cur_line['created']]
    # in case we wish to add ~60 communities which we manually labeled (most are communities which created for r/place)
    if add_handly_labeled_data:
        for com_name, values in handly_labeled_com.items():
            if com_name not in com_meta_data_dict:
                com_meta_data_dict[com_name] = [values['subscribers'], values['creation_epoch']]
    com_meta_data_df = pd.DataFrame.from_dict(data=com_meta_data_dict, orient='index')
    com_meta_data_df.reset_index(inplace=True)
    com_meta_data_df.columns = ['subreddit_name', 'number_of_subscribers', 'creation_epoch']
    return com_meta_data_df


def get_indices_inside_artwork(path, canvas_shape=(1000, 1000)):
    # function concept is taken from here -
    # https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
    x, y = np.meshgrid(np.arange(canvas_shape[0]), np.arange(canvas_shape[1]))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    try:
        p = Path(path)  # make a polygon
    # in case the path is empty
    except ValueError:
        return None
    grid = p.contains_points(points)
    mask = grid.reshape(canvas_shape[0], canvas_shape[1]).T
    pixels_inside_path = np.transpose(mask.nonzero())
    return pixels_inside_path


def mean_allocation_inside_polygon(indices_2d_array, placement_per_pixel,
                                   expand_grid_small_artworks=True, min_pixels=1000):
    if expand_grid_small_artworks:
        # pulling out the minimum and maximum pixels values (should be min of 0 and maximum of 1000)
        # we assume the min and max values would appear in the x-axis, no need to cover the y-axis as well
        min_axis = min([x_axis for x_axis, y_axis in placement_per_pixel.keys()])
        max_axis = max([x_axis for x_axis, y_axis in placement_per_pixel.keys()])
        indices_2d_array = expand_2d_grid(indices=indices_2d_array, minimum_indices_to_reach=min_pixels,
                                          min_axis=min_axis, max_axis=max_axis)
    # given a 2d array of indices - returns the average pixels allocations during the r/place experiment
    unknown_pixel_pair = 0
    allocations = list()
    for cur_pair in indices_2d_array:
        try:
            cur_pixel_tot_allocation = placement_per_pixel[cur_pair[0], cur_pair[1]]
            allocations.append(cur_pixel_tot_allocation)
        except KeyError:
            unknown_pixel_pair += 1
            continue
    return np.mean(allocations)


def expand_2d_grid(indices, minimum_indices_to_reach=1000, min_axis=0, max_axis=1000):
    # in case no expansion is needed or expansion is impossilbe (zero size matrix)
    if len(indices) >= minimum_indices_to_reach or len(indices) == 0:
        return indices
    cur_expanded_indices = indices.copy()
    cur_pixels_len = len(cur_expanded_indices)
    # we expand the pixels until we have enough pixels to return (controlled by the minimum_indices_to_reach arg)
    while cur_pixels_len < minimum_indices_to_reach:
        zero_column = np.zeros((cur_pixels_len, 1))
        one_column = np.ones((cur_pixels_len, 1))
        north_mask = cur_expanded_indices + np.hstack((zero_column, one_column))
        east_mask = cur_expanded_indices + np.hstack((one_column, zero_column))
        south_mask = cur_expanded_indices + np.hstack((zero_column, -one_column))
        west_mask = cur_expanded_indices + np.hstack((-one_column, zero_column))
        north_east_mask = cur_expanded_indices + np.hstack((one_column, one_column))
        south_east_mask = cur_expanded_indices + np.hstack((one_column, -one_column))
        south_west_mask = cur_expanded_indices + np.hstack((-one_column, -one_column))
        north_west_mask = cur_expanded_indices + np.hstack((-one_column, one_column))
        cur_expanded_indices = np.vstack((cur_expanded_indices, north_mask, east_mask, south_mask, west_mask,
                                          north_east_mask, south_east_mask, south_west_mask, north_west_mask))
        # removing duplications
        cur_expanded_indices = np.unique(cur_expanded_indices, axis=0)
        # remove non logical cases
        if min_axis is not None and max_axis is not None:
            cur_expanded_indices = \
                cur_expanded_indices[(cur_expanded_indices[:, 0] >= min_axis) & (cur_expanded_indices[:, 0] <= max_axis)
                                     & (cur_expanded_indices[:, 1] >= min_axis)
                                     & (cur_expanded_indices[:, 1] <= max_axis)].copy()
        # sorting the rows by first and second column
        ind = np.lexsort((cur_expanded_indices[:, 1], cur_expanded_indices[:, 0]))
        cur_expanded_indices = cur_expanded_indices[ind].copy()
        cur_pixels_len = len(cur_expanded_indices)
    return cur_expanded_indices


def is_artwork_fully_overlaps(atlas_mapping, big_artwork_idx, small_artwork_idx):
    small_artwork_mapping = atlas_mapping[atlas_mapping['atlas_idx'] == small_artwork_idx]
    small_artwork_mapping = small_artwork_mapping.iloc[0].to_dict()
    try:
        pixels_allocation_list = eval(small_artwork_mapping['overlaps with'])
    # in case the cell is empty, the eval will not work properly
    except TypeError:
        return False
    perc_of_big_in_small = [perc for artwork_idx, perc in pixels_allocation_list if artwork_idx == big_artwork_idx]
    if perc_of_big_in_small and perc_of_big_in_small[0] > 0.9:
        return True
    else:
        return False


def calc_artwork_area(path, atlas_idx, atlas_mapping):
    cur_artwork_in_atlas_mapping = atlas_mapping[atlas_mapping['atlas_idx'] == atlas_idx]
    cur_artwork_in_atlas_mapping = cur_artwork_in_atlas_mapping.iloc[0].to_dict()
    if cur_artwork_in_atlas_mapping['survived'] == 0:
        return 0, 0
    else:
        suggested_decrease_to_rainbowroad = 0
        cur_polygon = Polygon(path)
        cur_area = cur_polygon.area
        # in case the artwork has a conflict with another artwork - we will take care of it
        cur_pixels_allocation = eval(cur_artwork_in_atlas_mapping['pixels_allocation'])
        updated_area = cur_pixels_allocation[0][1] * cur_area
        # now we go over each artwork there is a conflict with and decide who if current artwork should get extra pixels
        for claim_artwork_idx, prop in cur_pixels_allocation[1:]:
            claim_artwork_survived = True if \
                atlas_mapping[atlas_mapping['atlas_idx'] == claim_artwork_idx].iloc[0]['survived'] == 1 else False
            # we give 100% of the area if the conflict is with artwork 286 (rainbow_road)
            if claim_artwork_idx == 286:
                updated_area += (cur_area * prop)
                suggested_decrease_to_rainbowroad += (cur_area * prop)
                continue
            # we also give 100% of the area if the conflict is with an artwork that did not survive
            elif not claim_artwork_survived:
                updated_area += (cur_area * prop)
                continue
            # we give 50% to the current artwork out of the total area we can assign - only in case there is not a huge
            # overlap between current artwork and the conflicted one. This is due to cases where 2 same communities
            # marked their success in the atlas with a full overlap
            # e.g., 'flag of the netherlands' and 'coat of arms of the netherlands'
            else:
                fully_overlap = is_artwork_fully_overlaps(atlas_mapping=atlas_mapping, big_artwork_idx=atlas_idx,
                                                          small_artwork_idx=claim_artwork_idx)
                updated_area += (0.5 * cur_area * prop) if not fully_overlap else 0
        return updated_area, suggested_decrease_to_rainbowroad


def calc_artwork_diameter(path, diametr_factor=1.0):
    x_diameter = max([i[0] for i in path]) - min([i[0] for i in path])
    y_diameter = max([i[1] for i in path]) - min([i[1] for i in path])
    diameter = max(x_diameter, y_diameter)
    return diameter * diametr_factor


def create_sr_name_to_id_dict(atlas_mapping):
    """
    maps between community name (sr_name) and an index of an artwork in the atlas.
    as an example: 'linux' community is associated with the index 0. Note that there might be more than one artwork
    index that the community can be associated with (e.g., 'de' is associated with 8 artworks)
    :param atlas_mapping: pandas df
        the big dataframe of artworks annotation (taken from the google sheet
    :return: dict
        a dictionary where ket is a string (name of the community) and value is a list of tuples. Each element in the
        list is a tuple of size 2 - index and percentage
    """
    exceptions = {'placenl': 'thenetherlands', 'placede': 'de', 'placecanada': 'canada'}
    sr_name_to_id_dict = defaultdict(list)
    for idx, cur_row in atlas_mapping.iterrows():
        cur_idx = cur_row['atlas_idx']
        cur_communities = cur_row['communities']
        # case the sr is empty (starts with '-'
        if cur_communities.startswith('-'):
            continue
        # case the sr is single (contains only a single community
        elif ',' not in cur_communities:
            cur_communities = cur_communities.lower().lstrip().rstrip()
            # converting the name of the community in case it is in the exceptions list (e.g., placede to de)
            cur_communities = cur_communities if cur_communities not in exceptions else exceptions[cur_communities]
            sr_name_to_id_dict[cur_communities].append((cur_idx, 1.0))
        # case multiple communities exist - then we give each a portion, as it is described in the table
        # for example: ['brazil', 'argentina', [1.0, 0.0]] means that only brazil will get the pixels
        else:
            communities_pixels_distrib = eval('['+cur_communities.split('[')[-1])
            num_srs = len(communities_pixels_distrib)
            cur_communities = cur_communities.split(',')[0:num_srs]
            cur_communities = [cc.lower().lstrip().rstrip() for cc in cur_communities]
            for portion, cc in zip(communities_pixels_distrib, cur_communities):
                # converting the name of the community in case it is in the exceptions list (e.g., placede to de)
                cc = cc if cc not in exceptions else exceptions[cc]
                # we will add information only in cases when portion > 0
                if portion > 0:
                    sr_name_to_id_dict[cc].append((cur_idx, portion))
    return sr_name_to_id_dict


def create_users_data_dict(csv_data_path, srs_to_pull_users_from, start_period='2017-01', end_period='2017-03'):
    # staring with the submissions
    submission_data = get_submissions_subset(files_path=csv_data_path,
                                             srs_to_include=None, start_month=start_period, end_month=end_period,
                                             min_utc=None, max_utc=None)
    users_to_include = set(submission_data[submission_data['subreddit'].str.lower().isin(set(srs_to_pull_users_from))]['author'])
    submission_data_filtered = submission_data[submission_data['author'].isin(users_to_include)]
    users_info = submission_data.groupby('author').agg({'author': 'size', 'score': 'sum',
                                                        'num_comments': 'sum', 'subreddit': 'nunique'})
    users_info.columns = ['submissions', 'submission_scores', 'comments_received', 'submissions_unique_srs']
    users_info_submissions_dict = users_info.to_dict(orient='index')

    del submission_data, users_info, users_to_include, submission_data_filtered
    gc.collect()

    # now doing the same for the comments
    comments_data = get_comments_subset(files_path=csv_data_path,
                                        srs_to_include=None, start_month=start_period, end_month=end_period,
                                        min_utc=None, max_utc=None)
    users_to_include = set(comments_data[comments_data['subreddit'].str.lower().isin(set(srs_to_pull_users_from))]['author'])
    comments_data_filtered = comments_data[comments_data['author'].isin(users_to_include)]
    users_info = comments_data.groupby('author').agg({'author': 'size',
                                                               'score': 'sum', 'subreddit': 'nunique'})
    users_info.columns = ['comments', 'comments_scores', 'comments_unique_srs']
    users_info_comments_dict = users_info.to_dict(orient='index')
    del comments_data, users_info, users_to_include, comments_data_filtered
    gc.collect()

    # updating both dicts with zero values
    for key in users_info_comments_dict.keys():
        if key in users_info_submissions_dict:
            continue
        else:
            users_info_submissions_dict[key] = {'submissions': 0, 'submission_scores': 0, 'comments_received': 0,
                                                'submissions_unique_srs': 0}
    for key in users_info_submissions_dict.keys():
        if key in users_info_comments_dict:
            continue
        else:
            users_info_comments_dict[key] = {'comments': 0, 'comments_scores': 0, 'comments_unique_srs': 0}
    joint_dict = {key: {**users_info_submissions_dict[key], **users_info_comments_dict[key]}
                  for key in users_info_comments_dict.keys()}
    return joint_dict


def calc_artwork_entropy(indices, pixels_at_end, smoothing=True):
    if len(indices) < 2:
        return 0
    indices_as_df = pd.DataFrame(indices)
    indices_as_df.columns = ['x_coordinate', 'y_coordinate']
    indices_as_df.set_index(['x_coordinate', 'y_coordinate'], inplace=True, drop=False)
    local_pixels_at_end = pixels_at_end.set_index(['x_coordinate', 'y_coordinate'], inplace=False).copy()
    merged_dfs = pd.merge(indices_as_df, local_pixels_at_end, how='inner', left_index=True, right_index=True)
    if merged_dfs.shape[0] < 2:
        return 0
    colors_mapping = dict(Counter(merged_dfs['color']))
    # in case the smoothing is turned on, we remove very rare pixles (those that have < 1% appearance)
    if smoothing:
        tot_pixels = sum(colors_mapping.values())
        colors_mapping_shrinked = {key: value for key, value in colors_mapping.items() if value/tot_pixels >= 0.01}
        artwork_ent = entropy(pd.DataFrame.from_dict(colors_mapping_shrinked, orient='index'))[0]
    else:
        artwork_ent = entropy(pd.DataFrame.from_dict(colors_mapping, orient='index'))[0]
    return artwork_ent


def is_new_com(com_name, com_meta_data, sr_obj_path='/data/work/data/reddit_place/sr_objects'):
    existing_com_names = [n.split('sr_obj_')[1].split('_.p')[0] for n in os.listdir(sr_obj_path)
                          if n.endswith('.p') and n.startswith('sr_obj_')]
    # checking if the community has an sr object
    if com_name in existing_com_names:
        com_is_new = False
        return com_is_new
    # if the name is not in the objects we created, we will check it in the meta data information
    cur_com_meta_data = com_meta_data[com_meta_data['subreddit_name'] == com_name]
    if cur_com_meta_data.shape[0] > 1:
        raise IOError("Error in the com_meta_data variable")
    # nothing was found
    elif cur_com_meta_data.shape[0] == 1:
        return True if cur_com_meta_data['creation_epoch'].values[0] >= 1490832000 else False
    # if the creation time is after +- the day in which r/place starter
    else:
        return None


def extract_liwc_occurrences(text_list, agg_results=True, liwc_file_path='/data/work/data/LIWC_Features.txt'):
    """
    Pull out a dictionary with the occurrences of the LIWC categories in a given text.
    It is based on a LIWC categories mapping given in the 'liwc_file_path
    :param text_list: list
        list of strings. Each item is a sentence. For each sentence we calculate the LIWC categories occurrences
    :param liwc_file_path: str
        the path to the LIWC text file. Default: location in yalla: '/data/work/data/LIWC_Features.txt'
    :param agg_results: boolean
        whether to aggregate the results of the LIWC categories occurrences
    :return: list
        list of size len(text_list). Each item is the LIWC categories occurrences in this piece of text.

    Examples
    --------
    >>> text_to_analyse = ['I myself am here', 'Great work. Good job', 'how ARE you?']
    >>> results = extract_liwc_occurrences(text_to_analyse)
    >>> print(results)

    """
    def _clean_text(local_text):
        if type(local_text) is not str:
            text = ''
        if local_text == '[deleted]' or local_text == '[removed]':
            local_text = ''
        deltabot_re = re.compile(r'^Confirmed: \d+ delta awarded to .*', re.DOTALL)
        if deltabot_re.match(local_text):
            text = ''
        local_text = local_text.lower()
        local_text = re.sub(r"what's", "what is ", local_text)
        local_text = re.sub(r"\'s", " ", local_text)
        local_text = re.sub(r"\'ve", " have ", local_text)
        local_text = re.sub(r"can't", "can not ", local_text)
        local_text = re.sub(r"n't", " not ", local_text)
        local_text = re.sub(r"i'm", "i am ", local_text)
        local_text = re.sub(r"\'re", " are ", local_text)
        local_text = re.sub(r"\'d", " would ", local_text)
        local_text = re.sub(r"\'ll", " will ", local_text)
        local_text = re.sub(r"\'scuse", " excuse ", local_text)
        mentions_re = re.compile(r'/u/\w*', re.MULTILINE)
        quote_re = re.compile(r'<quote>.[^<]*</quote>', re.MULTILINE)
        url_re = re.compile(r'http://[^\s]*', re.MULTILINE)
        for m in mentions_re.findall(local_text):
            local_text = local_text.replace(m, '_mention_')
        for q in quote_re.findall(local_text):
            local_text = local_text.replace(q, '_quote_')
        for url in url_re.findall(local_text):
            local_text = '_url_' + local_text.replace(url, tldextract.extract(url).domain)
        local_text = re.sub('\W', ' ', local_text)
        local_text = re.sub('\s+', ' ', local_text)
        local_text = local_text.strip(' ')
        return local_text

    # reading the LIWC dictionary
    if not os.path.isfile(liwc_file_path):
        raise IOError(f"The path given to the LIWC dict ({liwc_file_path}) is invalid")
    liwc_dict_path = open(liwc_file_path)
    lines = liwc_dict_path.readlines()
    liwc_dict_path.close()

    liwc_cat_dict = {}  # {cat: (w1,w2,w3,...)}
    for line in lines[1:]:  # first line is a comment about the use of *
        tokens = line.strip().lower().split(', ')
        liwc_cat_dict[tokens[0]] = tokens[1:]

    # creating a LIWC regex dict
    liwc_regex_dict = {}
    for k, v in liwc_cat_dict.items():
        s = '|'.join(v)
        s = re.sub(r'\*', r'\\w*', s)
        liwc_regex_dict[k] = re.compile(r'\b(' + s + r')\b')

    # handling the input text_list (text_list is a list of sentences
    cleaned_corpus = [_clean_text(local_text=t) for t in text_list]

    # creating a counter of the LIWC categories to the given list of texts
    liwc_categ_in_text = list()
    for cur_cc in cleaned_corpus:
        liwc_categ_in_text.append(Counter({cat: len(regex.findall(cur_cc))
                                           for cat, regex in liwc_regex_dict.items() if regex.findall(cur_cc)}))
    if agg_results:
        aggregated_res = defaultdict(int)
        # looping over each Counter object in the list
        for cur_res in liwc_categ_in_text:
            # looping over each category in the Counter (each is a LIWC category)
            for cur_categ, cnt in cur_res.items():
                aggregated_res[cur_categ] += cnt
        return aggregated_res
    else:
        return liwc_categ_in_text
