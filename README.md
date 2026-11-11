# Apartment Price Prediction

Predicting apartment sale prices from ~31,000 listings (Armenia, Nov-Dec 2022) with linear models
and tree-based models, evaluated with a leak-free train/test protocol.

## Problem
Given the characteristics of an apartment (area, region, renovation, building type, floor, ...),
predict its sale price in USD. The target is modeled on the log scale.

## Data
`data/apartments_for_sale.csv` - 17 columns, mixed currencies (USD, AMD, EUR, RUB) converted to USD with
approximate 2022 rates (`USD 1`, `AMD 1/400`, `RUB 1/65`, `EUR 1.05`). Rental listings were removed.

## Method
1. Cleaning: remove rentals and duplicates, convert currencies, fix impossible values.
2. EDA: missing values, price distribution, price per m2 by region, effects of listing characteristics.
3. Split first (80/20, stratified by region); all imputation and scaling happen inside a scikit-learn
   `Pipeline`, so nothing is learned from the test set.
4. Models: region-median baseline, Ridge, Random Forest, Gradient Boosting (randomized search, 5-fold CV).
5. Model chosen by cross-validation on the training set; the test set is used once.
6. Interpretation: permutation importance, partial dependence, depth-3 decision tree.
   
