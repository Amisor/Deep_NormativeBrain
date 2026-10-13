# A Deep Learning Framework for Mapping Normative Brain Development and Aging

The goal of this project is to use Quantile Regression via a Multi-Layer Perceptron to predict the 10th, 50th (median), and 90th percentiles of specific brain measurements given a subject's Age and Sex.

Instead of making a single point prediction, it estimates the variance and normative bounds of brain development, which is critical in psychiatric and neurological research for identifying individuals who deviate significantly from the population norm.

## Data

The Autism Brain Imaging Data Exchange (ABIDE) dataset provides previously collected resting state functional magnetic resonance imaging datasets from 539 individuals with ASD and 573 typical controls for the purpose of data sharing in the broader scientific community. 


* **ABIDE:** https://fcon_1000.projects.nitrc.org/indi/abide/
* **Phenotypic data:** http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
* **FreeSurfer outputs (ABIDE Preprocessed):** `https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/<FILE_ID>/stats/aseg.stats`
* **Download data:** [`notebooks/preprocessing_FC.ipynb`](notebooks/preprocessing_FC.ipynb), [`notebooks/preprocessing_structural.ipynb`](notebooks/preprocessing_structural.ipynb), [`notebooks/preprocessing_structural_site.ipynb`](notebooks/preprocessing_structural_site.ipynb)
* **Retrieved data:** [`data/`](data/) (processed datasets and scaling parameters in [`data/processed/`](data/processed/))

### Preprocessing

* `preprocessing_FC`: downloads the ABIDE fMRI ROI time series (Harvard-Oxford atlas, 111 regions) of the healthy controls with Nilearn, and computes the Functional Connectivity (FC) between ROI 0 and ROI 1 (Pearson correlation, Fisher z-transformed).
* `preprocessing_structural`: downloads the FreeSurfer `aseg.stats` file of each healthy control that passed the anatomical quality control, and extracts the total gray matter volume (`TotalGrayVol`), intracranial volume and hippocampus volumes.
* `preprocessing_structural_site`: same as `preprocessing_structural`, plus the acquisition site as one-hot columns.

All preprocessing notebooks z-score Age and the target, and save the scaling parameters to transform the predictions back to original units.

## Model

`QuantileMLP`: a Multi-Layer Perceptron with a shared backbone and three outputs, the 10th, 50th and 90th percentiles.

**Non-crossing quantiles.** If each percentile is predicted independently, the network can end up with $q_{10}$ larger than $q_{50}$ (quantile crossing), which is not possible. To avoid it, the model predicts the median and two distances, made positive with Softplus:

$$\hat{y}_{10} = \hat{y}_{50} - \text{Softplus}(\Delta_{10}) \qquad \hat{y}_{90} = \hat{y}_{50} + \text{Softplus}(\Delta_{90})$$

where $\text{Softplus}(x) = \log(1 + e^x)$ is always positive, so $\hat{y}_{10} \le \hat{y}_{50} \le \hat{y}_{90}$ always.

**Pinball loss.** Each output learns its percentile with the pinball (quantile) loss, for a quantile $\tau$ and error $u = y - \hat{y}_{\tau}$:

$$L_{\tau}(u) = \max(\tau \, u, \; (\tau - 1) \, u)$$

For $q_{10}$, predicting too high costs 9 times more than too low, so about 10% of the subjects end up below it (and the same idea for $q_{50}$ and $q_{90}$).

**Training.** 70/15/15 train/validation/test split, the model with the lowest validation loss is kept. The model is compared with a baseline that ignores the inputs (same percentiles for everyone), and checked with the test set calibration (% of subjects below each percentile).

### Methods

All models use the same `QuantileMLP`, pinball loss and training loop. They differ in the target and the inputs.

* `model_FC` (Age, Sex → FC): weak age effect, the model barely improves over the baseline.
* `model_structural` (Age, Sex → `TotalGrayVol`): clear age and sex effects, but volumes differ a lot between sites (scanners), so the test set is poorly calibrated.
* `model_structural_site` (Age, Sex, Site → `TotalGrayVol`): adding the site about doubles the improvement over the baseline.
* `model_structural_site_balanced` (Age, Sex, Site → `TotalGrayVol`): same as `model_structural_site`, with the split balanced by site. Best calibration. Includes growth charts per site and for all subjects adjusted to a reference site.

