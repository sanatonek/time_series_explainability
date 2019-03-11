# time_series_explainability

Explaining the output of time series perdiction models in healthcare

## Run the following scripts to query and preprocess the ICU mrtality data (This step might take a few hours)
`
python3 data_generator/icu_mortality.py ---sqluser YOUR_USER --sqlpass YOUR_PASSWORD
python3 data_generator/data_preprocess.py
`
## Once the data is ready, train models using TSX/main.py
