from os.path import join as opj
import pickle
import pandas as pd
import sys
if sys.platform == 'linux':
    sys.path.append('/sise/home/isabrah/reddit_canvas/reddit_project_with_yalla_cluster/reddit-tools')
from r_place_success_analysis.data_creation.data_creation_utils import *

# configurations
data_path = "/sise/Yalla_work/data/reddit_place/canvas_annotation_effort_data"
smooth_demand_area_per_artwork = False
min_pixels_per_area_for_demanding_calc = 1000
save_target_df = True
target_df_saving_path = opj(data_path, 'success_analysis_target_df_25_08_2022.p')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

if __name__ == "__main__":
    # data loading
    # list of dicts, taken from the atlas data source. Each entry is an artwork
    atlas_data_as_list = pickle.load(open(opj(data_path, 'atlas_data_as_list.p'), "rb"))
    # a data-frame, taken from the work Or did to map art-works to communities using the atlas data
    atlas_mapping = pd.read_csv(opj(data_path, 'atlas_mapping.csv'))
    # a data-frame, with the location, color and user names of all tiles placement in the experiment
    tiles_placement = pd.read_csv(opj(data_path, 'tiles_with_explicit_user_names.csv'))
    placement_per_pixel = tiles_placement[['x_coordinate', 'y_coordinate']].groupby(['x_coordinate', 'y_coordinate']).size()
    # creating a df to hold the find color in each cell at the end of the process
    pixels_at_end = tiles_placement.loc[tiles_placement.groupby(['x_coordinate', 'y_coordinate'])["ts"].idxmax()].copy()
    pixels_at_end = pixels_at_end[['x_coordinate', 'y_coordinate', 'color']].copy()
    # converting it to a dict
    placement_per_pixel = placement_per_pixel.to_dict()
    # meta data information per community - in order to know if the community is new or not (per r/place starting time)
    # and to know the community size
    com_meta_data = load_com_meta_data(data_path='/sise/Yalla_work/data/reddit_place')

    # creating the basic target features per community (which left a mark on the canvas), that is:
    # a. area (number of pixels).
    # b. difficulty level in the area of the artwork (how attractive was the area where the picture was drawn.
    # c. diameter (from upper left corner to bottom right one).
    target_feature_artwork_level = dict()
    tot_suggested_decrease_to_rainbowroad = 0
    # looping over each item (artwork) in the atlas list and calculating its overall area
    for cur_artwork in atlas_data_as_list:
        cur_atlas_idx = cur_artwork['id']
        # identifying the relevant row in the atlas_mapping - only if the artwork survived we "care" about it
        cur_artwork_info_dict = atlas_mapping[atlas_mapping['atlas_idx'] == cur_atlas_idx].iloc[0].to_dict()
        cur_artwork_survived = cur_artwork_info_dict['survived']
        indices_inside_artwork = get_indices_inside_artwork(cur_artwork['path'])
        # in case the artwork did not survive OR is has an empty area (empty path)
        if indices_inside_artwork is None or cur_artwork_survived == 0:
            continue
        # in addition to the overall pixels allocation, we also calculate a normalized metric, which takes into account
        # the total number of pixels that were placed during the experiment
        cur_mean_allocations = \
            mean_allocation_inside_polygon(indices_inside_artwork, placement_per_pixel,
                                           expand_grid_small_artworks=True if smooth_demand_area_per_artwork else False,
                                           min_pixels=min_pixels_per_area_for_demanding_calc)
        cur_artwork_area, decrease_to_rainbowroad = calc_artwork_area(path=cur_artwork['path'], atlas_idx=cur_atlas_idx,
                                                                      atlas_mapping=atlas_mapping)
        cur_artwork_diameter = \
            calc_artwork_diameter(cur_artwork['path'],
                                  diametr_factor=eval(cur_artwork_info_dict['pixels_allocation'])[0][1])
        cur_artwork_ent = calc_artwork_entropy(indices_inside_artwork, pixels_at_end)
        tot_suggested_decrease_to_rainbowroad += decrease_to_rainbowroad
        target_feature_artwork_level[cur_atlas_idx] = {'area': cur_artwork_area, 'mean_demand': cur_mean_allocations,
                                                       'diameter': cur_artwork_diameter, 'entropy': cur_artwork_ent}
    # end of loop
    # updating the rainbowraod number of pixels
    target_feature_artwork_level[286]['area'] = target_feature_artwork_level[286]['area'] - \
                                                tot_suggested_decrease_to_rainbowroad
    # let's  see how many pixels we allocate in total (should be ~1M)
    print(f"The total amount of pixels allocation over the whole canvas is: "
          f"{sum([value['area'] for key, value in target_feature_artwork_level.items()])} (it should be close to 1M)")
    # in order to see the sorted list of pixels allocation
    area_dict = {key: value['area'] for key, value in target_feature_artwork_level.items()}
    sorted_list_of_pixels = sorted(area_dict.items(), key=lambda item: item[1], reverse=True)
    print(f"Here is the list of the top-10 artworks and their total pixels amount:\n {sorted_list_of_pixels[0:10]}")

    # up to now we had the mapping per artwork. However, we want to hav a mapping per each community
    # for that we will use the mapping we did between artwork, communities and pixels
    target_feature_com_level = dict()
    sr_name_to_id_dict = create_sr_name_to_id_dict(atlas_mapping=atlas_mapping)
    # looping over each item in the dict created (over each community)
    for cur_sr_name, artworks_mapping in sr_name_to_id_dict.items():
        cur_tot_pixels = 0
        cur_tot_diameter = 0
        # the demand and the entropy are a bit more complex, so we save it as a list and later handle it
        # (we need to know the total pixels per community to calculate both correctly)
        cur_tot_demand_list = list()
        cur_tot_entropy_list = list()
        # looping over each item in the artworks associated with the current sr-
        for (artwork_idx, percentage) in artworks_mapping:
            pixels_to_add = target_feature_artwork_level[artwork_idx]['area'] * \
                            percentage if artwork_idx in target_feature_artwork_level else 0
            cur_tot_pixels += pixels_to_add
            cur_tot_diameter += target_feature_artwork_level[artwork_idx]['diameter'] * \
                                percentage if artwork_idx in target_feature_artwork_level else 0
            demand_value_to_add = target_feature_artwork_level[artwork_idx]['mean_demand'] * \
                                percentage if artwork_idx in target_feature_artwork_level else 0
            cur_tot_demand_list.append((demand_value_to_add, pixels_to_add))
            entropy_value_to_add = target_feature_artwork_level[artwork_idx]['entropy'] * \
                                   percentage if artwork_idx in target_feature_artwork_level else 0
            cur_tot_entropy_list.append((entropy_value_to_add, pixels_to_add))
        # now we should have all values per community to create the dict.
        # The demand and the entropy are the most complicated one - we calculate it using weighted mean
        mean_demand = np.sum([ctdl[0] * (ctdl[1] / cur_tot_pixels)
                              for ctdl in cur_tot_demand_list]) if cur_tot_pixels > 0 else 0
        mean_entropy = np.sum([ctel[0] * (ctel[1] / cur_tot_pixels)
                              for ctel in cur_tot_entropy_list]) if cur_tot_pixels > 0 else 0
        # we also include information whether the community is a new one (created only for the r/place experiment or not
        is_new = is_new_com(com_name=cur_sr_name, com_meta_data=com_meta_data,
                            sr_obj_path='/sise/Yalla_work/data/reddit_place/sr_objects')
        try:
            com_size = com_meta_data[com_meta_data['subreddit_name'] == cur_sr_name]['number_of_subscribers'].values[0]
        except IndexError:
            com_size = 0
        target_feature_com_level[cur_sr_name] = {'tot_pixels': cur_tot_pixels,
                                                 'tot_diameter': cur_tot_diameter,
                                                 'mean_demand': mean_demand,
                                                 'mean_entropy': mean_entropy,
                                                 'is_created_for_rplace': is_new,
                                                 'community_size': com_size,
                                                 'survived': True if cur_tot_pixels > 0 else False,
                                                 'manually_labeled': True
                                                 }
    # adding the extinct communities based on the historical data we used (WWW paper) + manual annotation
    r_place_com_mapping = pd.read_csv(opj('/sise/Yalla_work/data/reddit_place/subreddits_rplace.csv'))
    all_revealed_com = list(r_place_com_mapping[r_place_com_mapping['rplace'] == 1]['name'])
    extinct_com_names_model_labeled = [arc for arc in all_revealed_com if arc not in target_feature_com_level]
    extinct_com_df = pd.read_csv(opj(data_path, 'extinct_communities.csv'))
    extinct_com_manually_labeled = list(extinct_com_df[extinct_com_df['distinct_effort'] == 1]['Community'])

    for cur_sr_name in extinct_com_names_model_labeled:
        is_new = is_new_com(com_name=cur_sr_name, com_meta_data=com_meta_data,
                            sr_obj_path='/sise/Yalla_work/data/reddit_place/sr_objects')
        try:
            com_size = com_meta_data[com_meta_data['subreddit_name'] == cur_sr_name]['number_of_subscribers'].values[0]
        except IndexError:
            com_size = 0
        if cur_sr_name not in target_feature_com_level:
            target_feature_com_level[cur_sr_name] = {'tot_pixels': 0, 'tot_diameter': 0,
                                                     'mean_demand': 0, 'survived': False,
                                                     'mean_entropy': 0,
                                                     'is_created_for_rplace': is_new,
                                                     'community_size': com_size,
                                                     'manually_labeled': True if cur_sr_name in
                                                                                 extinct_com_manually_labeled else False
                                                     }
    com_level_df = pd.DataFrame.from_dict(target_feature_com_level, orient='index')
    # short analysis part
    sr_name_to_total_pixels = sorted(com_level_df.tot_pixels.to_dict().items(), key=lambda x: x[1], reverse=True)
    print(f"List of the top-20 communities and their total pixels allocation:\n {sr_name_to_total_pixels[0:20]}")
    sr_name_to_avg_demand = sorted(com_level_df.mean_demand.to_dict().items(), key=lambda x: x[1], reverse=True)
    print(f"\nList of the top-20 communities in regards to the mean pixels demand:\n {sr_name_to_avg_demand[0:20]}")
    sr_name_to_avg_entropy = sorted(com_level_df.mean_entropy.to_dict().items(), key=lambda x: x[1], reverse=True)
    print(f"\nList of the top-20 communities in regards to their artwork entropy:\n {sr_name_to_avg_entropy[0:20]}")
    if save_target_df:
        pickle.dump(com_level_df, open(target_df_saving_path, "wb"))
        print(f"\nA gold label df of size {com_level_df.shape} was created and saved under {target_df_saving_path}")
        print("\nNow you can move to create the modeling df using a file called 'modeling_df_creation.py'. Good Luck!")

    """
    LeftOvers
    # simple correlation against the continuous feature
    print(modeling_df.corr()['overall_pixels'])
    print(modeling_df.corr('spearman')['overall_pixels'])

    # average value of explanatory features VS the multi-class feature
    print(modeling_df.groupby('success_level').mean().transpose())

    # point bi-serial correlation against the binary feature
    explanatory_feature_to_analyse = 'centrality_avg'  # 'comments_amount'#'days_pazam'#'users_amount'
    a = ma.masked_invalid(modeling_df[explanatory_feature_to_analyse])
    b = ma.masked_invalid(modeling_df['is_successful'])
    msk = (~a.mask & ~b.mask)
    print(pointbiserialr(modeling_df[msk][explanatory_feature_to_analyse], modeling_df[msk]['is_successful']))
    """

