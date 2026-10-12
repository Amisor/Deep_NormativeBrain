# A Deep Learning Framework for Mapping Normative Brain Development and Aging

The goal of this project is to use Quantile Regression via Deep Learning model to predict the 10th, 50th (median), and 90th percentiles of specific brain measurements given a subject's Age and Sex.

Instead of making a single point prediction, it estimates the variance and normative bounds of brain development, which is critical in psychiatric and neurological research for identifying individuals who deviate significantly from the population norm.

## Data

The Autism Brain Imaging Data Exchange (ABIDE) dataset provides previously collected resting state functional magnetic resonance imaging datasets from 539 individuals with ASD and 573 typical controls for the purpose of data sharing in the broader scientific community. 


ABIDE: https://fcon_1000.projects.nitrc.org/indi/abide/
Download data: ../notebooks/preprocessing


Phenotypic data: http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html

All the retreived data from ABIDE is available at ../data
