# Car price prediction

This school project is about predicting second hand cars using machine learning.
This project needed skill in Machine learning (linear regression, NLP), data cleaning, and feature engineering.
Files :
- Main files are at the roots of the repo
- EDA is in the `notebook` folder
- `autopluspy` is a custom python library made for this project

## Getting started

1. git clone the project
2. create a virtualenv
```bash
virtualenv -p python3 venv
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Put the initial dataset into `/data` folder

5. Run the jupyter notebook Runbook (available at the roots of the repo) to launch the whole system. 
Uncomment the last cell if you want to start the streamlit app

## Architecture and features

### Data Engineering
Input:
- Initial dataset
- Eventually new dataset 

Process:
- [X] Spot and remove duplicated content (rows and columns) 
- [X] Spot and remove missing values
- [X] Adapt data type (categorical, numerical, datetime, string)
- [X] Provide insight about unique value for each categorical value
- [X] Provide insight about each numerical value (.describe())
- [X] Get dummies of categorical variable in One Hot Encoder (update Data Dictionary)
- [X] Compute age of the car (Online - Model Year)
- [X] Count vectorizer on 'Options:'

- [ ] Scrap AutoPlus and fuzzy match
- [ ] Use Data Mapper

Output
- Processed dataset
- Data Dictionary

### Machine Learning
Input:
- Dataset
- Data Dictionary

process:
- [X] Learn object: 
    - Original dataset in df / dataset.original /return df
    - Train split /dataset.train_set/ return df X_train, y_train
    - Test split /dataset.test_set /return df X_test, y_test
    - Data Dictionary /dataset.data_dictionary /return df
- [X] Analyze target variable distribution
- [X] Normalization of numerical features
- [X] Analyze features variance
- [ ] Multi collinearity handling https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
- Feature selection)
- [X] CV
- [X] Grid search and select best score based on CV results
- [X] Results on test set
- [ ]Train on full learn_set
- [ ] SHAP/LIME/permutation_importance interpretation
- [X] Performance metrics : MAPE

Output:
- Regression Model
- Std model 
- Features needed for prediction with possible value

### App

Input:
- Data Dictionary
- Features list needed for the prediction

Interaction :
- [X] form
- [X] display prediction and price tuning range
- [ ] how this car price is considering others cars price (good deal or not)

### Cheatsheet Streamlit

```python3
## User input : text
Model_year = st.text_input('Model_year', '2010')
```

