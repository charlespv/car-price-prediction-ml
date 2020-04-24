# Car price prediction

This school project is about predicting second hand cars using machine learning.
This project need skill in Machine learning (linear regression, NLP), data cleaning, feature engineering and model serving.

## Roadmap
1. Make a first prediction pipeline without "Description" column and very few column. The model resulting to this pipeline will be the first ground truth to beat.
2. Implement model serving app with Streamlit and add it to pipeline
2. Make a classical cleaning stage, run the pipeline
3. Make a classical preprocessing stage, run the pipeline
4. Make a stage to handle "Description" columns
4. Benchmark few models (example : linreg, rf, lightgbm, small neural network for the fun)
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
- Remove duplicated content (rows and columns)
- Remove missing values
- Adapt data type (categorical, numerical, datetime, string)
- Provide insight about unique value for each categorical value
- Provide insight about each numerical value (.describe())
- Encode Categorical variable in One Hot Encoder (update Data Dictionary)
- Extract features from "Description"

Output
- Processed dataset
- Data Dictionary

### Machine Learning
Input:
- Dataset
- Data Dictionary

process:
- learn object: 
    - Original dataset in df / dataset.original /return df
    - Train split /dataset.train_set/ return df X_train, y_train
    - Test split /dataset.test_set /return df X_test, y_test
    - Data Dictionary /dataset.data_dictionary /return df
- analyze target variable distribution
- normalization
- feature selection (corr matrix, RFE, Sharp with basic model)
- grid search and select best score based on CV results
- results on test set
- Train on full learn_set

Output:
- model 
- features needed for prediction with possible value

### App