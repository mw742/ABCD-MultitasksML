# ABCD-MultitasksML

**Multi-Task Machine Learning of Environmental and Genetic Risk Pathways to Substance Use Initiation in the ABCD Study**

Mengman Wei*, Hanwen Zhang, and Qian Peng*


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

If you use this repository in your research, please cite it as follows:

Wei, M., Zhang, H., & Peng, Q. (2025). Multi-Task Machine Learning of Environmental and Genetic Risk Pathways to Substance Use Initiation in the ABCD Study. Retrieved from https://github.com/mw742/ABCD-MultitasksML

