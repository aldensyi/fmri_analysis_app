import nibabel as nib
import numpy as np
import nilearn
import pandas as pd
import json
import os
from os.path import join
from nilearn.plotting import plot_img
from nilearn.image import mean_img
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn.plotting import plot_contrast_matrix
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map
from nilearn.reporting import get_clusters_table

def load_file(inputfile):
    """
    Loading the nifti file format and header information

        Parameters:
            inputfile (str): the file path name within the system

        Returns:
            img(FileBasedImage): Nifti image object
            hdr(FileBasedHeader): Header object of the inputted Nifti image
    """
    img = nib.load(inputfile)
    hdr = img.header
    return img, hdr

# For Debugging Purposes
def generate_first_slice(img_obj, save:bool, save_add):
    """
    Generating and storing the first slice of the BOLD functional MRI
    
        Parameters:
            img_obj(FileBasedImage): Nifti image object
            save(bool): Option to save
            save_add: Local path address to save the image at
    """
    func_data = nilearn.image.index_img(img_obj, 0)
    plot_img(func_data, colorbar=True, cbar_tick_format="%i")
    
    if save:
        final_save_add = save_add + "images/first_slice.png"
        plt.savefig(final_save_add)
        plt.close()

def generate_glm_firstLevel(img_obj, hdr_obj, eventfile, param):
    """
    Generating First Level GLM
    
        Parameters:
            img_obj(FileBasedImage): Nifti image object
            hdr_obj(FileBasedHeader): Nifti header object
            eventfile(Dataframe): CSV file containing the details of the 
                                  conditional periods of the experiment
            param(Dict): dictionary object containing the customized parameters
                                  for the GLM analysis

        Returns:
            fmri_glm(FirstLevelModel): GLM for the said session
    """

    fmri_glm = FirstLevelModel(
        t_r=int(hdr_obj.get_zooms()[3]),
        noise_model=param["noise_model"],
        standardize=param["standardize"],
        hrf_model=param["hrf_model"],
        drift_model=param["drift_model"],
        high_pass=param["high_pass"],
        
        slice_time_ref = param["slice_time_ref"],
        drift_order = param["drift_order"],
        fir_delays = param["fir_delays"],
        min_onset = param["min_onset"],
        mask_img = param["mask_img"],
        smoothing_fwhm = param["smoothing_fwhm"],
        verbose = param["verbose"],
        subject_label = param["subject_label"] 
    )
    
    fmri_glm = fmri_glm.fit(img_obj, eventfile)

    return fmri_glm

def get_design_mtrx(glm_model_object, save: bool, save_add):
    """
    Generating and storing Design Matrix
    
        Parameters:
            glm_model_object(FirstLevelModel): GLM object created from the Nifti
            image
            save(bool): option to save
            save_add(str): file path address to save instance of design matrix
    
        Returns:
            design_matrix(Dataframe): Design Matrix from the GLM object that was
            created earlier
    """

    design_matrix = glm_model_object.design_matrices_[0]

    plot_design_matrix(design_matrix)
    
    if save:
        final_save_add = save_add + "images/design_matrix.png"
        plt.savefig(final_save_add)

    plt.close()

    return design_matrix

def get_conditions(design_matrix_df: pd.DataFrame):
    """
    Extracting the Conditions from the previously created Design Matrix

        Parameters:
            design_matrix_df(Dataframe): Design Matrix from a General Linear Model
        
        Returns:
            conditions(List): a list containing just the conditions from the
            Dataframe
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
    Generating the Expected Response Graph for the condition that was passed in
    through the "conditional" argument
    
        Parameters:
            design_mtrx(Dataframe): Design Matrix from a General Linear Model
            conditional(str): The condition label from the experiment
            save(bool): Boolean to save the design matrix as a jpeg instance
            sv_add(str): local file path address to save the jpeg
    """

    plt.plot(design_mtrx[conditional])
    plt.xlabel("scan")
    plt.title(f"Expected Response for Condition: \"{conditional}\"")
    
    if save:
        final_save_add = sv_add + f"images/Expected_Response_{conditional}.jpeg"
        plt.savefig(final_save_add)
        plt.close()

def create_contrast_matrix_instance(design_matrix_df: pd.DataFrame):
    """
        Creating contrast matrix via extracting the length of the design matrix
        and assigning true values based on the index of the condition within the
         dictionary

        Parameters:
            design_matrix_df(Dataframe): Design Matrix from a General Linear Model
    
        Returns:
            conditions(dict): dict object containing conditions of the NifTi 
            object and its corresponding matrix
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

def plotting_contrast_matrices(cond_dict: dict, design_matrx, save: bool, save_add):
    """
    Generates contrasts matrixs of the conditions that were passed in through 
    via the conditions dictionary created previously

        Parameters:
            cond_dict(dict): dict object containing conditions of the NifTi
            object and its corresponding matrix
            design_matrx(Dataframe): Design Matrix from a General Linear Model
            save(Bool): Boolean to save the contrast matrix/ces as a jpeg instance
            save_add(str): Local file path address to save the jpeg instance
    """

    for conds in cond_dict:
        plot_contrast_matrix(cond_dict[conds], design_matrix=design_matrx)
        
        if save:
            final_save_add = save_add + f"images/contrast_matrix_{conds}"
            plt.savefig(final_save_add)
            plt.close()

def get_threshold_map_z(zmap, background_img, thrshold: int, condtn:str="", save:bool=False, save_add:str=""):
    """
    Generates a visualization of the statistical map with voxels that are have
    z-scores greater than the user-set threshold and is diplayed in the z plane
    
        Parameters:
            zmap(Nifti Image): Z-scaled Statistical Map formed from t-statistic
            background_img(Nifti Image): Mean Background functional image
            thrshold(int): Z-score threshold
            condtn(str): Condition Label
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance

    """
    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=thrshold,
        display_mode="z",
        black_bg=True,
        title=f"{condtn} (Z>3)"
    )

    if save:
        final_save_add = save_add + f"images/stat_map_zscore_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()



def get_threshold_map_p(zmap, background_img, a_val: float=0.001, condtn:str="", save:bool=False, save_add:str=""):
    """
    Generates a visualization of the statistical map with voxels that have 
    p-values less than the user-set alpha-value and is diplayed in the z plane
    
        Parameters:
            zmap(Nifti Image): Z-scaled Statistical Map formed from t-statistic
            background_img(Nifti Image): Mean Background functional image
            a_val(int): false positive rate threshold
            condtn(str): Condition Label
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance
    
    """

    output_map, threshld = threshold_stats_img(zmap, alpha=a_val, height_control="fpr")

    output_map.to_filename(join(save_add, f"p_stat/p-score_thresholded_map_{condtn}.nii.gz"))
    
    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        black_bg=True,
        title=f"{condtn} (p<{a_val})"
    )

    if save:
        final_save_add = save_add + f"images/stat_map_pscore_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def get_threshold_map_p_bonferroni(zmap, background_img, a_val: float=0.05, condtn:str="", save:bool=False, save_add:str=""):
    """
    Generates a visualization of the statistical map with voxels that are Bonferroni-corrected,
    has a p-value of less than 0.05, and is diplayed in the z plane
    
        Parameters:
            zmap(Nifti Image): Z-scaled Statistical Map formed from t-statistic
            background_img(Nifti Image): Mean Background functional image
            a_val(int): false positive rate threshold
            condtn(str): Condition Label
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance
    
    """

    output_map, threshld = threshold_stats_img(zmap, alpha=a_val, height_control="bonferroni")

    output_map.to_filename(join(save_add, f"p_stat_bonferroni_corrected/bonferroni-corrected_thresholded_map_{condtn}.nii.gz"))

    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        black_bg=True,
        title=f"{condtn} (p<{a_val}, corrected)" 
    )

    if save:
        final_save_add = save_add + f"images/stat_map_bonferroni_corrected_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def get_threshold_map_fdr(zmap, background_img, a_val: float=0.05,  condtn:str="", save:bool=False, save_add:str=""):    
    """
    Generates a visualization of the statistical map with voxels that are 
    corrected via a set False Discovery Rate 
    
        Parameters:
            zmap(Nifti Image): Z-scaled Statistical Map formed from t-statistic
            background_img(Nifti Image): Mean Background functional image
            a_val(int): false positive rate threshold
            condtn(str): Condition Label
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance
    
    """

    output_map, threshld = threshold_stats_img(zmap, alpha=a_val, height_control="fdr")

    output_map.to_filename(join(save_add, f"p_stat_fdr_corrected/fdr-corrected_thresholded_map_{condtn}.nii.gz"))

    plot_stat_map(
        zmap,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        black_bg=True,
        title=f"{condtn} (fdr={a_val})" 
    )

    if save:
        final_save_add = save_add + f"images/stat_map_fdr_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()

def get_large_cluster_threshold_map(zmap, background_img, a_val: float=0.05, clstr:int=10, condtn:str="", save:bool=False, save_add:str=""):
    """
    Generates a visualization of the statistical map with activated significant 
    voxel clusters that are corrected via a set False Discovery Rate
    
        Parameters:
            zmap(Nifti Image): Z-scaled Statistical Map formed from t-statistic
            background_img(Nifti Image): Mean Background functional image
            a_val(int): false positive rate threshold
            clstr: Cluster size threshold (cluster size below threshold will not be shown)
            condtn(str): Condition Label
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance
    
    """

    clean_map, threshld = threshold_stats_img(
        zmap, alpha=a_val, height_control="fdr", cluster_threshold=clstr
    )

    clean_map.to_filename(join(save_add, f"large_cluster_thresholded/large_cluster_thresholded_map_{condtn}.nii.gz"))

    plot_stat_map(
        clean_map,
        bg_img=background_img,
        threshold=threshld,
        display_mode="z",
        black_bg=True,
        title=f"{condtn} (fdr={a_val}), clusters > {clstr} voxels" 
    )

    if save:
        final_save_add = save_add + f"images/stat_map_large_clusters_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def get_clusters_threshold_table(zmap, a_val: float=0.05, clstr:int=10, condtn:str="", save:bool=False, save_add:str=""):
    """
    - Generating a csv/table version of the Clusters Threshold Map
    
        Parameters:
            zmap(Nifti Image): Z-scaled Statistical Map formed from t-statistic
            a_val(int): false positive rate threshold
            clstr: Cluster size threshold (cluster size below threshold will not be shown)
            condtn(str): Condition Label
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance
    
    """

    _, threshld = threshold_stats_img(
        zmap, alpha=a_val, height_control="fdr", cluster_threshold=clstr
    )
    
    table = get_clusters_table(
        zmap, stat_threshold=threshld, cluster_threshold=clstr
    )

    if save:
        table.to_csv(rf"{save_add}clusters/clusters_table_{condtn}.csv",sep=",", header=True)

def get_effects_of_interest_matrix(conditions: dict, dsgn_mtrx, save:bool, save_add:str):
    """
    Generates a Effects of Interest contrast matrix based on all the conditions
      that were part of the NifTi file

        Parameters:
            conditions(dict): Contrast Matrix of the conditions
            dsgn_mtrx(Dataframe): Design Matrix from a GLM
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance

    """

    e_o_i= np.vstack(tuple(conditions.values()))

    plot_contrast_matrix(e_o_i, dsgn_mtrx)

    if save:
        final_save_add = save_add + f"images/contrast_matrix_effect_of_interest.jpeg"
        plt.savefig(final_save_add)
        plt.close()

def get_f_test_map(zmap, background_img, a_val: float=0.05, clstr:int=10, condtn:str="",save:bool=False, save_add:str=""):
    """
    Generate a visualization of the statistical map of the F test in which one 
    seeks whether a certain combination of conditions (possibly two-, three- 
    or higher-dimensional), explaining a significant proportion of the signal.
    
        Parameters:
            zmap(Nifti Image): Z-scaled Statistical Map formed from t-statistic
            background_img(Nifti Image): Mean Background functional image
            a_val(int): false positive rate threshold
            clstr: Cluster size threshold (cluster size below threshold will not be shown)
            condtn(str): Condition Label
            save(bool): Boolean on whether or not to save the plot map as a jpeg
            save_add(str): Local file path address to save the jpeg instance

    """

    clean_map, threshold = threshold_stats_img(
        zmap, alpha=a_val, height_control="fdr", cluster_threshold=clstr
    )

    clean_map.to_filename(join(save_add, f"f_test/f-test_map_{condtn}.nii.gz"))

    plot_stat_map(
        clean_map,
        bg_img=background_img,
        threshold=threshold,
        display_mode="z",
        black_bg=True,
        title=f"Effects of interest for {condtn}(fdr=0.05), clusters > 10 voxels",
    )
    
    if save:
        final_save_add = save_add + f"images/stat_map_zscore_{condtn}.jpeg"
        plt.savefig(final_save_add)
        plt.close()


def main():
    # Loading config.json values
    with open('config.json') as f:
        config = json.load(f)

    # Create directories
    directories = ['images', 'clusters', 'z_stat', 'p_stat', 'p_stat_bonferroni_corrected', 'p_stat_fdr_corrected', 'large_cluster_thresholded', 'f_test']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Attaining the img and the header object
    subj_img, subj_hdr = load_file(config["bold"])
    
    # Use when checking for header information
    '''
    data_shape = subj_hdr.get_data_shape()
    voxel_sizes = subj_hdr.get_zooms()
    spatial_units, temporal_units = subj_hdr.get_xyzt_units()

    print("Data shape:", data_shape)
    print("Voxel sizes:", voxel_sizes)
    print("Spatial units:", spatial_units)
    print("Temporal units:", temporal_units)
    '''

    # Importing the events file and assigning commonly used objects within the config.json file as a local object
    evnts = pd.read_table(config["events"])
    save_path = "./"
    glm_param = {}

    # putting all the GLM Parameters into the glm_param dict object
    glm_param["noise_model"] = config["noise_model"]
    glm_param["standardize"] = config["standardize"]
    glm_param["hrf_model"] = config["hrf_model"]
    glm_param["high_pass"] = config["high_pass"]
    glm_param["slice_time_ref"] = config["slice_time_ref"]
    glm_param["drift_order"] = config["drift_order"]
    glm_param["min_onset"] = config["min_onset"]
    glm_param["verbose"] = config["verbose"]
    glm_param["subject_label"] = config["subject_label"]

    if config["smoothing_fwhm"]:
        glm_param["smoothing_fwhm"] = config["smoothing_fwhm"]
    else:
        glm_param["smoothing_fwhm"] = None
    if config["drift_model"] != "None":
        glm_param["drift_model"] = config["drift_model"]
    else:
        glm_param["drift_model"] = None

    if "mask_img" in config:
        if config["mask_img"]:
            glm_param["mask_img"] = config["mask_img"]
        else:
            glm_param["mask_img"] = None

    # Generates first slice and saves it; Commented out for debugging use
    # generate_first_slice(subj_img, True, save_path)
    
    # Generates mean slice
    mean_bg_img = mean_img(subj_img)

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
    subj_contrst_mtrx = create_contrast_matrix_instance(subj_dsgn_mtrx)

    # Saving the Contrast Matrix as an media file
    plotting_contrast_matrices(subj_contrst_mtrx, subj_dsgn_mtrx,True, save_path)
    
    # Traversing each condition available and getting all the analysis out and storing them within the previously created directories
    for item in subj_contrst_mtrx:
        eff_map_subj = subj_glm.compute_contrast(subj_contrst_mtrx[item], output_type="effect_size")
        z_map_subj = subj_glm.compute_contrast(subj_contrst_mtrx[item], output_type="z_score")
        z_map_subj.to_filename(join(save_path, f"z_stat/z_map_{item}.nii.gz"))
        
        get_threshold_map_z(z_map_subj, mean_bg_img, 3, item, True, save_path)

        get_threshold_map_p(z_map_subj, mean_bg_img, 0.001, item, True, save_path)

        get_threshold_map_p_bonferroni(z_map_subj, mean_bg_img, 0.05, item, True, save_path)

        get_threshold_map_fdr(z_map_subj, mean_bg_img, 0.05, item, True, save_path)

        get_large_cluster_threshold_map(z_map_subj, mean_bg_img, 0.05, 10, item, True, save_path)

        get_clusters_threshold_table(z_map_subj, 0.05, 10, item, True, save_path)

        get_f_test_map(z_map_subj, mean_bg_img, 0.05, 10, item, True, save_path)
        
    # Generating Effect of Interest contrast matrix and the f-test statistical map as QA measurements
    get_effects_of_interest_matrix(subj_contrst_mtrx, subj_dsgn_mtrx, True, save_path)

   


if __name__ == "__main__":
    main()
