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