# Kaggle Competition 1 - Extreme Weather Events
This repository contains the work realized by Thierry Jean in the context of the graduate course IFT 6390 - Machine Learning Fundamentals of Fall 2021.

## Competition description
A subset of the ClimateNet dataset containing 47,760 labelled weather events with 18 variables for meteorological events labelled as either: *standard conditions* (class 0), *tropical cyclone* (class 1), or *atmospheric river* (class 2). For this task, several multiclass classifiers will be trained, including *Softmax multiclass logistic regression*, *One-vs-one binary logistic regression*, and *XGBoost*. The performance of the models will be evaluated using a test set of 7,320 unlabeled events.

## Approaches tried
- Sampling, cross-validation, and preprocessing functions were implemented from scratch
- Softmax classifier was implemented from scratch
- One-versus-one binary logistic regression ensemble was implemented from scratch
- Gradient boosting machine using xgboost

## How to run the code
The code for this competition is found in **comp1_dev.ipynb**. All cells should be runned in linear order. The notebook is structured using Markdown headers hierarchy, which can be folded to view and hide sections.  It is organized as follow:
- Imports
- Data load
- Exploration
- Data preparation
- Model
- Producing stats
- Export results
- Compare submissions

## Packages used
- numpy
- pandas
- matplotlib
- xgboost
- sklearn