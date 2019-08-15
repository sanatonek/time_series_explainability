# time_series_explainability

Explaining the output of time series perdiction model

## Data preparation
This model is evaluated on 3 different datasets. Follow these steps to prepare data for experiment
# MIMIC ICU dataset:
Run the following scripts to query and preprocess the ICU mrtality data (This step might take a few hours)
```
python3 data_generator/icu_mortality.py ---sqluser YOUR_USER --sqlpass YOUR_PASSWORD
```
```
python3 data_generator/data_preprocess.py
```
# Simulated dataset:

## Model training
First train the predictor model to explain
```
python3 -m TSX.main --model risk_predictor --train
```
Then run other explainer classes on the trained model
