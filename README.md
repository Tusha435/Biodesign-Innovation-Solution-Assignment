# Colon Cancer Analysis and Classification

This repository provides an end-to-end workflow for analyzing, preprocessing, and building classification models on colon cancer datasets. We use Python libraries such as `pandas`, `matplotlib`, `seaborn`, `sklearn`, and `shap` for data manipulation, visualization, machine learning, and feature importance interpretation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Data Loading and Exploration](#data-loading-and-exploration)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Train-Test Split](#train-test-split)
7. [Model Training](#model-training)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Model Evaluation](#model-evaluation)
10. [Feature Importance](#feature-importance)
11. [SHAP Analysis](#shap-analysis)

## Project Overview
This project performs an analysis and classification on colon cancer data. The workflow includes:
1. Loading and exploring two separate datasets.
2. Merging the datasets based on `Type of Colon Cancer`.
3. Encoding categorical variables and scaling numeric features.
4. Engineering risk and severity indices.
5. Training and evaluating Logistic Regression and Random Forest models.
6. Visualizing feature importance using SHAP.

## Requirements
Install the required packages:
```bash
pip install pandas matplotlib seaborn scikit-learn shap

## Data Loading and Exploration
import pandas as pd

# Load the datasets
dataset_1 = pd.read_csv('dataset_1_colon_cancer.csv')
dataset_2 = pd.read_csv('dataset_2_colon_cancer.csv')

# Display first few rows and basic info
dataset_1_head = dataset_1.head()
dataset_1_info = dataset_1.info()

dataset_2_head = dataset_2.head()
dataset_2_info = dataset_2.info()
