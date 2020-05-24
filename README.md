# Car price prediction

This school project is about predicting second hand cars using machine learning.
This project need skill in Machine learning (linear regression, NLP), data cleaning, feature engineering and model serving.

## Roadmap
1. Make a first prediction pipeline without "Description" column and very few column. The model resulting to this pipeline will be the first ground truth to beat.
2. Implement model serving app with Streamlit and add it to pipeline
2. Make a classical cleaning stage, run the pipeline
3. Make a classical preprocessing stage, run the pipeline
4. Make a stage to handle "Description" columns
4. Benchmark few models (example : linreg, rf)
5. Bonus : fetch new dataset to improve the model

## Getting started

1. git clone the project
2. create a virtualenv
```bash
virtualenv -p python3 venv
```
3. Install dependencies
```bash
pip install requirements.txt
```
4. Put the initial dataset into `/data` folder

5. Run the app
```bash
streamlit run app.py
```

## Architecture and features

### Data Engineering
Input:
- Initial dataset
- Eventually new  dataset 

Process:
- [X] Spot and remove duplicated content (rows and columns) 
- [X] Spot and remove missing values
- [X] Adapt data type (categorical, numerical, datetime, string)
- [X] Provide insight about unique value for each categorical value
- [X] Provide insight about each numerical value (.describe())
- [X] Get dummies of categorical variable in One Hot Encoder (update Data Dictionary)
- [X] Compute age of the car (Online - Model Year)

- [ ] Count vectorizer on 'Options:'
- [ ] Use Data Mapper
- [ ] Scrap AutoPlus and fuzzy match

Output
- Processed dataset
- Data Dictionary

### Machine Learning
Input:
- Dataset
- Data Dictionary

process:
- Learn object: 
    - Original dataset in df / dataset.original /return df
    - Train split /dataset.train_set/ return df X_train, y_train
    - Test split /dataset.test_set /return df X_test, y_test
    - Data Dictionary /dataset.data_dictionary /return df
- Analyze target variable distribution
- Normalization of numerical features
- Analyze features variance
- Multi collinearity handling https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
- Feature selection)
- CV
- Grid search and select best score based on CV results
- Results on test set
- Train on full learn_set
- SHAP/LIME/permutation_importance interpretation
- Performance metrics : MAPE, MAE

Output:
- Regression Model
- Std model 
- Features needed for prediction with possible value

### App

Input:
- Data Dictionary
- Features list needed for the prediction

Interaction :
- form
- display prediction and price tuning range
- how this car price is considering others cars price (good deal or not)

### Cheatsheet Streamlit

```python3
## User input : text
Model_year = st.text_input('Model_year', '2010')
```

