<h1 align="center">First Level GLM Analysis</h1>

<p align="center">
  Fits a first-level GLM to a single-session task fMRI run and its events with Nilearn FirstLevelModel, producing z, F and thresholded (uncorrected, Bonferroni, FDR, cluster) statistical maps.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.808"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.808-blue.svg" alt="Run on brainlife.io"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Fits a first-level GLM to a single-session task fMRI run and its events with Nilearn FirstLevelModel, producing z, F and thresholded (uncorrected, Bonferroni, FDR, cluster) statistical maps.**

Adapted from Nilearn's single-subject GLM tutorial. It builds the design matrix from the events file with the chosen HRF and drift models, fits the GLM (optionally within a mask), computes per-condition contrasts and writes z maps, p-thresholded maps with uncorrected, Bonferroni and FDR control, large-cluster maps, F-test maps, cluster tables and QA images. Use it for single-subject task activation maps from a BOLD run with a BIDS events.tsv.

| | |
|---|---|
| **Inputs** | `neuro/func/task` · `neuro/mask` (optional) |
| **Outputs** | `neuro/maps` ×6 · `generic/images` · `neuro/contrast-stats` |
| **Container** | `brainlife/ga-python:lab328-dipy141-pybrainlife-1.0` |
| **Run it** | [doi.org/10.25663/brainlife.app.808](https://doi.org/10.25663/brainlife.app.808) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `bold_fmri` | `neuro/func/task` | yes | Task BOLD fMRI run with its events file. events.tsv is required to build the design matrix. |
| `brain_mask_optional` | `neuro/mask` | no | Mask to be used on data. If an instance of masker is passed, then its mask will be used. If no mask is given, it will be computed automatically by a NiftiMasker with default parameters. |

#### Outputs

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

<details>
<summary><b>Parameters</b> (11)</summary>

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

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.808](https://doi.org/10.25663/brainlife.app.808), choose `neuro/func/task`, `neuro/mask` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/maps`, `neuro/maps`, `neuro/maps`, `neuro/maps`, `neuro/maps`, `neuro/maps`, `generic/images`, `neuro/contrast-stats`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/ga-python:lab328-dipy141-pybrainlife-1.0`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "bold": "/path/to/bold_fmri/bold",
       "events": "/path/to/bold_fmri/events",
       "events_json": "/path/to/bold_fmri/events_json",
       "sbref": "/path/to/bold_fmri/sbref",
       "sbref_json": "/path/to/bold_fmri/sbref_json",
       "physio": "/path/to/bold_fmri/physio",
       "physio_json": "/path/to/bold_fmri/physio_json",
       "mask_img": "/path/to/brain_mask_optional/mask_img",
       "noise_model": "ar1",
       "standardize": false,
       "hrf_model": "glover",
       "drift_model": "cosine",
       "high_pass": 0.01,
       "slice_time_ref": 0,
       "drift_order": 1,
       "min_onset": -24,
       "smoothing_fwhm": 0,
       "subject_label": "",
       "verbose": 1
   }
   ```

3. Run `./main`.

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- First Level GLM Analysis. brainlife.io App. [doi:10.25663/brainlife.app.808](https://doi.org/10.25663/brainlife.app.808)
- Garyfallidis et al. Dipy, a library for the analysis of diffusion MRI data. *Frontiers in Neuroinformatics* 8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Abraham et al. Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics* 8 (2014). [doi:10.3389/fninf.2014.00014](https://doi.org/10.3389/fninf.2014.00014)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Sangmin Yi

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io, which runs this App, is publicly funded by Wellcome, the U.S. National Institutes of Health and the U.S. National Science Foundation; acknowledging that support helps keep it running.</sub>
