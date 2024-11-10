# Colon Cancer Prediction Analysis
This project analyzes two datasets on colon cancer, focusing on exploratory data analysis, data preprocessing, feature engineering, and predictive modeling using Logistic Regression and Random Forest. This README provides a step-by-step guide to each part of the code and its functionality.

## Table of Content
1. Project Overview
2. Dependencies
3. Data Loading and Initial Exploration
4. Data Visualization
5. Data Merging
6. Data Preprocessing
7. Feature Engineering
8. Train-Test Split
9. Model Training
10. Hyperparameter Tuning
11. Model Evaluation
12. Feature Importance Analysis
13. Explainability using SHAP

## 1. Project Overview
The project aims to identify important factors and build predictive models for colon cancer diagnosis based on patients' clinical data and tumor characteristics. We use two datasets, dataset_1 and dataset_2, which contain both numerical and categorical features, for a comprehensive analysis and modeling process.

## 2. Dependencies
-Pandas
-matplotlib
-seaborn
-sklearn
-shap

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

