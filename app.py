import nibabel as nib
import numpy as np
import nilearn
import pandas as pd
import json
import os
from nilearn.plotting import plot_img
from nilearn.image import mean_img
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn.plotting import plot_contrast_matrix
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map
from nilearn.reporting import get_clusters_table

def loadFile(inputfile):
    """
    - Loading the nifti file format and header information
    - Usually will be accessing the config.json file for the inputfile
    - Returns a image object and header object
    """
    
    img = nib.load(inputfile)
    hdr = img.header
    return img, hdr

def checkHeader(hdr_obj):
    """
    - Checking the header information to see if there are any empty fields
    - Mostly for debugging purposes
    """
    
    data_shape = hdr_obj.get_data_shape()
    voxel_sizes = hdr_obj.get_zooms()
    spatial_units, temporal_units = hdr_obj.get_xyzt_units()

    print("Data shape:", data_shape)
    print("Voxel sizes:", voxel_sizes)
    print("Spatial units:", spatial_units)
    print("Temporal units:", temporal_units)

def generate_first_slice(img_obj, save:bool, save_add):
    """
    - Generating the first slice of the BOLD functional MRI
    """
    func_data = nilearn.image.index_img(img_obj, 0)
    plot_img(func_data, colorbar=True, cbar_tick_format="%i")
    
    if save:
        final_save_add = save_add + "slices/first_slice.png"
        plt.savefig(final_save_add)
        plt.close()
    
def get_mean_slice(img_obj):
    """
    - Generating the mean slice of the object using BOLD functional MRI image object
    - Returns a mean nifti image
    """
    return mean_img(img_obj)

def get_stimfile(inputfile):
    """
    - Getting tsv/csv file containing the events for the specific trial
    - Returns a dataframe
    """
    return pd.read_table(inputfile)

def generate_glm_firstLevel(img_obj, hdr_obj, eventfile, param):
    """
    - Generating First Level GLM
    - param will be a dictionary object containing the user-set parameter for the GLM
    - Returns a GLM
    """

    fmri_glm = FirstLevelModel(
        t_r=int(hdr_obj.get_zooms()[3]),
        noise_model=param["noise_model"],
        standardize=param["standardize"],
        hrf_model=param["hrf_model"],
        drift_model=param["drift_model"],
        high_pass=param["high_pass"]
    )
    
    fmri_glm = fmri_glm.fit(img_obj, eventfile)

    return fmri_glm

def get_design_mtrx(glm_model_object, save: bool, save_add):
    """
    - Generating Design Matrix and save when user set save boolean to true
    - Returns GLM
    """

    design_matrix = glm_model_object.design_matrices_[0]

    plot_design_matrix(design_matrix)
    
    if save:
        final_save_add = save_add + "matrix/design_matrix.png"
        plt.savefig(final_save_add)

    plt.close()

    return design_matrix

def get_conditions(design_matrix_df: pd.DataFrame):
    """
    - Extracting the Conditions from the previously created Design Matrix
    - Returns a List object containing just the conditions from the Dataframe
    """
    
    keywords = ["drift_", "constant"]
    conditions = []
    column_hdrs = design_matrix_df.columns
    
    for element in column_hdrs:
        if keywords[0] not in element and keywords[1] not in element:
            conditions.append(element)
    return conditions

def display_expected_response_graph(design_mtrx, conditional, save: bool, sv_add):
    """
    - Generating the Expected Response Graph for the condition that was passed in through the "conditional" argument
    """

    plt.plot(design_mtrx[conditional])
    plt.xlabel("scan")
    plt.title(f"Expected Response for Condition: \"{conditional}\"")
    
    if save:
        final_save_add = sv_add + f"graphs/Expected_Response_{conditional}.jpeg"
        plt.savefig(final_save_add)
        plt.close()

def create_contrast_mtrix(design_matrix_df: pd.DataFrame):
    """
    - Creating contrast matrix via extracting the length of the design matrix and assigning true values based on the index of the condition within the dictionary
    - Returns dictionary containing conditions of the NifTi object and its corresponding matrix
    """
    
    # getting the column length of design matrix 
    col_len = len(design_matrix_df.columns)
    conditions = {}

    # Removes drifts and constants from the condition columns
    main_conditions = get_conditions(design_matrix_df)

    # creating contrast matrix containing all zeros (all falses)
    for elements in main_conditions:
        conditions[elements] = np.zeros(col_len)

    # assigning true values based on position
    key_list = list(conditions.keys())

    for x in range(len(conditions)):
        conditions[key_list[x]][x] = 1

    return conditions

def plot_contrst_matrix(cond_dict: dict, design_matrx, save: bool, save_add):
    """
    - Generates contrasts matrixs of the conditions that were passed in through via the conditions dictionary created previously
    """

    for conds in cond_dict:
        plot_contrast_matrix(cond_dict[conds], design_matrix=design_matrx)
        
        if save:
            final_save_add = save_add + f"matrix/contrast_matrix_{conds}"
            plt.savefig(final_save_add)
            plt.close()


def get_effmap(conds, input_glm):
    """
    - conds: condition that needs a effect map; needs to the np.array
    - input glm: fmri_glm object created earlier
    """
    return input_glm.compute_contrast(conds, output_type="effect_size")

def get_zmap(conds, input_glm):
    """
    - conds: condition that needs a effect map; needs to the np.array
    - input glm: fmri_glm object created earlier
    """
    return input_glm.compute_contrast(conds, output_type="z_score")


def get_threshold_map_z(zmap, background_img, thrshold: int, cut_crds=3, condtn:str="", save:bool=False, save_add:str=""):
    """
    - Generates a visualization of the statistical map with voxels that are have z-scores greater than the user-set threshold and is diplayed in the z plane
    """
    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=thrshold,
        display_mode="z",
        cut_coords=cut_crds,
        black_bg=True,
        title=f"{condtn} (Z>3)"
    )

    if save:
        final_save_add = save_add + f"maps/stat_map_zscore_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def get_threshold_map_p(zmap, background_img, a_val: float=0.001, cut_crds=3, condtn:str="", save:bool=False, save_add:str=""):
    """
    - Generates a visualization of the statistical map with voxels that have p-values less than the user-set alpha-value and is diplayed in the z plane
    - a_val can be adjusted
    """

    _, threshld = threshold_stats_img(zmap, alpha=a_val, height_control="fpr")
    
    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        cut_coords=cut_crds,
        black_bg=True,
        title=f"{condtn} (p<{a_val})"
    )

    if save:
        final_save_add = save_add + f"maps/stat_map_pscore_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def get_threshold_map_p_bonferroni(zmap, background_img, a_val: float=0.05, cut_crds=3, condtn:str="", save:bool=False, save_add:str=""):
    """
    - Generates a visualization of the statistical map with voxels that are Bonferroni-corrected, has a p-value of less than 0.05 and is diplayed in the z plane
    - a_val can be adjusted
    """

    _, threshld = threshold_stats_img(zmap, alpha=a_val, height_control="bonferroni")

    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        cut_coords=cut_crds,
        black_bg=True,
        title=f"{condtn} (p<{a_val}, corrected)" 
    )

    if save:
        final_save_add = save_add + f"maps/stat_map_bonferroni_corrected_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def get_threshold_map_fdr(zmap, background_img, a_val: float=0.05, cut_crds=3, condtn:str="", save:bool=False, save_add:str=""):    
    """
    - Generates a visualization of the statistical map with voxels that are corrected via a set False Discovery Rate 
        - set False Discovery Rate: Controlling the expected proportion of false discoveries among detections by setting it to 0.05 or 5%
        - The base rate can be set by the user
    """

    _, threshld = threshold_stats_img(zmap, alpha=a_val, height_control="fdr")

    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        cut_coords=cut_crds,
        black_bg=True,
        title=f"{condtn} (fdr={a_val})" 
    )

    if save:
        final_save_add = save_add + f"maps/stat_map_fdr_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()

def get_large_cluster_threshold_map(zmap, background_img, a_val: float=0.05, clstr:int=10,cut_crds=3, condtn:str="", save:bool=False, save_add:str=""):
    """
    - Generates a visualization of the statistical map with activated significant voxel clusters that are corrected via a set False Discovery Rate
    - a_val can be adjusted for the False Discovery Rate
    """

    clean_map, threshld = threshold_stats_img(
        zmap, alpha=a_val, height_control="fdr", cluster_threshold=clstr
    )

    plot_stat_map(
        clean_map,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        cut_coords=cut_crds,
        black_bg=True,
        title=f"{condtn} (fdr={a_val}), clusters > {clstr} voxels" 
    )

    if save:
        final_save_add = save_add + f"maps/stat_map_large_clusters_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def get_clstrs_table(zmap, a_val: float=0.05, clstr:int=10, condtn:str="", save:bool=False, save_add:str=""):
    """
    - Generating a csv/table version of the Clusters Threshold Map
    - a_val (alpha value) and the cluster limit (base set to 10 or higher voxels) can be adjusted by the user
    """

    _, threshld = threshold_stats_img(
        zmap, alpha=a_val, height_control="fdr", cluster_threshold=clstr
    )
    
    table = get_clusters_table(
        zmap, stat_threshold=threshld, cluster_threshold=clstr
    )

    if save:
        table.to_csv(rf"{save_add}table/clusters_table_{condtn}.csv",sep=",", header=True)

"""
Done with Analysis

Now for QA and f-test
"""


def effcts_interest(conditions: dict, dsgn_mtrx, save:bool, save_add:str):
    """
    - Generates a Effects of Interest contrast matrix based on all the conditions that were part of the NifTi file
    """

    e_o_i= np.vstack(tuple(conditions.values()))

    plot_contrast_matrix(e_o_i, dsgn_mtrx)

    if save:
        final_save_add = save_add + f"matrix/contrast_matrix_effect_of_interest.jpeg"
        plt.savefig(final_save_add)
        plt.close()

def get_ftest_map(zmap, background_img, a_val: float=0.05, clstr:int=10,cut_crds=3, condtn:str="",save:bool=False, save_add:str=""):
    """
    - Generate a visualization of the statistical map of the F test in which one seeks whether a certain combination of conditions (possibly two-, three- or higher-dimensional),
    which explains a significant proportion of the signal.
    """

    clean_map, threshold = threshold_stats_img(
        zmap, alpha=a_val, height_control="fdr", cluster_threshold=clstr
    )

    plot_stat_map(
        clean_map,
        bg_img=background_img,
        threshold=threshold,
        display_mode="z",
        cut_coords=cut_crds,
        black_bg=True,
        title=f"Effects of interest for {condtn}(fdr=0.05), clusters > 10 voxels",
    )
    
    if save:
        final_save_add = save_add + f"maps/stat_map_zscore_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def main():
    # Loading config.json values
    with open('config.json') as f:
        config = json.load(f)

    # Create directories
    directories = ['data', 'data/slices', 'data/matrix', 'data/table', 'data/graphs', 'data/maps']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Attaining the img and the header object
    subj_img, subj_hdr = loadFile(config["bold"])
    
    # Use when checking for header information
    """
    data_shape = subj_hdr.get_data_shape()
    voxel_sizes = subj_hdr.get_zooms()
    spatial_units, temporal_units = subj_hdr.get_xyzt_units()

    print("Data shape:", data_shape)
    print("Voxel sizes:", voxel_sizes)
    print("Spatial units:", spatial_units)
    print("Temporal units:", temporal_units)
    """

    # Importing the events file and assigning commonly used objects within the config.json file as a local object
    evnts = get_stimfile(config["events"])
    save_path = config["save_directory"]
    glm_param = config["glm_parameters"]

    # Generates first slice and saves it
    generate_first_slice(subj_img, True, save_path)
    
    # Generates mean slice
    mean_bg_img = get_mean_slice(subj_img)

    # Generates GLM Object
    subj_glm = generate_glm_firstLevel(subj_img, subj_hdr, evnts, glm_param)

    # Generates design matrix
    subj_dsgn_mtrx = get_design_mtrx(subj_glm, True, save_path)

    # Retrieving the conditions from the session
    subj_conds = get_conditions(subj_dsgn_mtrx)

    # Displaying Expected Response Graph for each individual condition available
    for individual_cond in subj_conds:
        display_expected_response_graph(subj_dsgn_mtrx, individual_cond, True, save_path)
    
    # Generate Contrast Matrix
    subj_contrst_mtrx = create_contrast_mtrix(subj_dsgn_mtrx)

    # Saving the Contrast Matrix as an media file
    plot_contrst_matrix(subj_contrst_mtrx, subj_dsgn_mtrx,True, save_path)
    
    # Traversing each condition available and getting all the analysis out and storing them within the previously created directories
    for item in subj_contrst_mtrx:
        eff_map_subj = get_effmap(subj_contrst_mtrx[item], subj_glm)
        z_map_subj = get_zmap(subj_contrst_mtrx[item], subj_glm)
        
        get_threshold_map_z(z_map_subj, mean_bg_img, 3, 3, item, True, save_path)

        get_threshold_map_p(z_map_subj, mean_bg_img, 0.001, 3, item, True, save_path)

        get_threshold_map_p_bonferroni(z_map_subj, mean_bg_img, 0.05, 3, item, True, save_path)

        get_threshold_map_fdr(z_map_subj, mean_bg_img, 0.05, 3, item, True, save_path)

        get_large_cluster_threshold_map(z_map_subj, mean_bg_img, 0.05, 10, 3, item, True, save_path)

        get_clstrs_table(z_map_subj, 0.05, 10, item, True, save_path)

    # Generating Effect of Interest contrast matrix and the f-test statistical map as QA measurements
    effcts_interest(subj_contrst_mtrx, subj_dsgn_mtrx, True, save_path)

    get_ftest_map(z_map_subj, mean_bg_img, 0.05, 10, 3, item, True, save_path)


if __name__ == "__main__":
    main()