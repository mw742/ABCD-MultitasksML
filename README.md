# ABCD-MultitasksML

**Multi-Task Machine Learning of Environmental and Genetic Risk Pathways to Substance Use Initiation in the ABCD Study**

Mengman Wei, Hanwen Zhang, and Qian Peng


## Overview
This repository contains code for performing multi-task machine learning (MTL) to predict substance use initiation (alcohol, nicotine, cannabis, and any substance) based on baseline environmental and genetic risk factors from the ABCD Study® (Adolescent Brain Cognitive Development). The goal is to model and predict the onset of substance use initiation using MTL to capture shared structures and domain-specific risks.


## Scripts
- **Data Preprocessing**: Conversion of time-to-event phenotype data into binary initiation labels for four substances (alcohol, nicotine, cannabis, and any substance) based on a fixed 48-month horizon.
- **Train/Validation/Test Splits**: Subject-level splits with zero overlap across datasets.
- **Multi-Task Model**: A neural network with a shared trunk and task-specific heads, optimized using masked binary cross-entropy loss and class weights.
- **Baseline Models**: Single-task logistic regression models using L2 and ElasticNet regularization for comparison.
- **Hyperparameter Optimization**: Utilizes Optuna for hyperparameter tuning.
- **Performance Metrics**: Evaluates models based on AUROC, AUCPR, and Brier score.


## Dependencies
python/3.11.4
pytorch/2.5.0py311-cuda


## Citation

If you use this work, please cite:

Wei, M., Zhang, H., & Peng, Q. (2026). *Dynamic and Baseline Multi-Task Learning for Predicting Substance Use Initiation in the ABCD Study*. medRxiv. doi: 10.64898/2026.04.10.26350655.


## Data access and responsible use

This repository provides analysis code and does not include participant-level ABCD Study data. Users must obtain authorized access to the required data through the appropriate official provider before running the analyses. Data use must comply with the applicable data-use agreements, approved research purposes, and institutional requirements.

The software license applies to the code and does not grant access to ABCD Study data or permission to redistribute external datasets or restricted derived outputs.

Generated files may contain participant identifiers, longitudinal measurements, genetic scores, substance-use outcomes, or other sensitive information. Do not upload these files to public repositories or share them with unauthorized individuals. Any sharing must comply with the applicable data-use terms and required approvals.

For reproducibility, document the cohort, data release/version, accession or DOI where available, preprocessing steps, software versions, and exact code commit used. Include acknowledgments and citations required by the data provider.

## Version status and updates

The scripts currently posted may reflect an earlier development version. Later revisions may exist outside this repository, and the correspondence between the posted scripts and the code used for the reported analyses has not yet been verified.

The review conducted to date included source-code and repository-history inspection, Python syntax checks, and limited synthetic preprocessing checks. It did not include execution of the R pipelines, neural-network training, end-to-end execution, or independent reproduction of the associated study results. Identified issues apply to the reviewed repository version; their relevance to later versions and their effects on the reported results have not been established. Undetected errors or limitations may remain.

The posted scripts will be checked against available later versions and analysis records, with corrections made where needed. Updates will be documented in the commit history or release notes, including the validation performed and any known effects on outputs.

Users should record the exact commit used and independently assess the code’s suitability for their intended analyses.
