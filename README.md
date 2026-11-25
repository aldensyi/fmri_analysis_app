<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

**Fits a first-level GLM to a single-session task fMRI run and its events with Nilearn FirstLevelModel, producing z, F and thresholded (uncorrected, Bonferroni, FDR, cluster) statistical maps.**

Adapted from Nilearn's single-subject GLM tutorial. It builds the design matrix from the events file with the chosen HRF and drift models, fits the GLM (optionally within a mask), computes per-condition contrasts and writes z maps, p-thresholded maps with uncorrected, Bonferroni and FDR control, large-cluster maps, F-test maps, cluster tables and QA images. Use it for single-subject task activation maps from a BOLD run with a BIDS events.tsv.

Run it on brainlife.io: [https://doi.org/10.25663/brainlife.app.808](https://doi.org/10.25663/brainlife.app.808)

| Input | Datatype | Required | Description |
|---|---|---|---|
| `bold_fmri` | `neuro/func/task` | yes | Task BOLD fMRI run with its events file. events.tsv is required to build the design matrix. |
| `brain_mask_optional` | `neuro/mask` | no | Mask to be used on data. If an instance of masker is passed, then its mask will be used. If no mask is given, it will be computed automatically by a NiftiMasker with default parameters. |

| Output | Datatype | Description |
|---|---|---|
| `f_test` | `neuro/maps` | F-test statistical maps per condition. |
| `large_cluster_thresholded` | `neuro/maps` | Thresholded maps keeping only large clusters. |
| `p_stat` | `neuro/maps` | Maps thresholded at uncorrected p (false positive rate). |
| `p_stat_bonferroni_corrected` | `neuro/maps` | Maps thresholded with Bonferroni correction. |
| `p_stat_fdr_corrected` | `neuro/maps` | Maps thresholded with FDR correction. |
| `z_stat` | `neuro/maps` | Z-score maps per condition contrast. |
| `images` | `generic/images` | Captured PNGs of the graphs, stat-maps and the matrices produced during the processes |
| `clusters` | `neuro/contrast-stats` | Cluster tables (CSV) of thresholded maps. |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `noise_model` | enum | `ar1` | The temporal variance model Options: `ar1`, `ols`. |
| `standardize` | boolean | `false` | If standardize is True, the time-series are centered and normed: their variance is put to 1 in the time dimension. |
| `hrf_model` | enum | `glover` | This parameter defines the HRF model to be used. Options: `spm`, `spm + derivative`, `spm + derivative + dispersion`, `glover`, `glover + derivative`, `glover + derivative + dispersion`, `fir`. |
| `drift_model` | enum | `cosine` | This parameter specifies the desired drift model for the design matrices. Options: `polynomial`, `cosine`, `None`. |
| `high_pass` | number | `0.01` | This parameter specifies the cut frequency of the high-pass filter in Hz for the design matrices. Used only if drift_model is ‘cosine’ |
| `slice_time_ref` | number | `0` | This parameter indicates the time of the reference slice used in the slice timing preprocessing step of the experimental runs. It is expressed as a fraction of the t_r (time repetition), so it can have values between 0. and 1. |
| `drift_order` | number | `1` | This parameter specifies the order of the drift model (in case it is polynomial) for the design matrices. |
| `min_onset` | number | `-24` | This parameter specifies the minimal onset relative to the design (in seconds). Events that start before (slice_time_ref * t_r + min_onset) are not considered. |
| `smoothing_fwhm` | number | `0` | If smoothing_fwhm is not None, it gives the full-width at half maximum in millimeters of the spatial smoothing to apply to the signal. |
| `subject_label` | string |  | Subject label passed to Nilearn FirstLevelModel. |
| `verbose` | number | `1` | Verbosity level of Nilearn FirstLevelModel. |

Runtime: container `brainlife/ga-python:lab328-dipy141-pybrainlife-1.0`.

<!-- brainlife-app:end -->

[![brainlife.io/app](https://img.shields.io/badge/brainlife.io-app-green.svg)](https://brainlife.io/app/65a9a68c4ce5ac2907f02c52)

### Authors
- [Sangmin Yi](sminyi0130@gmail.com)

### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your publications and code reusing this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)

### Citations
There is not a particular paper published concerning the module from Nilearn and its modules.

However, if you want to cite Nilearn and its related modules:
```json
@software{Nilearn,
    author = {Nilearn contributors},
    license = {BSD-4-Clause},
    title = {{nilearn}},
    url = {https://github.com/nilearn/nilearn},
    doi = {https://doi.org/10.5281/zenodo.8397156}
}  
```

#### MIT Copyright (c) 2024 brainlife.io The University of Texas at Austin

## Running the App
