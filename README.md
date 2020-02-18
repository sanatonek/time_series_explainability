# Instance-wise Feature Importance in Time (FIT)

Explaining the output of time series perdiction model

## Data preparation
Three different datasets are used in the experiments. The process of creating or preprocessing the data is explained below.


### Simulated dataset (State data):
Run the following script to create the data and the ground thruth explanations for the state experiment. You can choose the total number of samples in the dataset as well as the lenght of eachrecording. The deafualts are set to 1000 samples of length 100.
```
python3 data_generator/state_data.py --signal_len LENGTH_OF_SIGNALS --signal_num TOTAL_NUMBER_OF_SAMPLES
```

### Simulated dataset (Spike data):
```
python3 data_generator/simulations_threshold_spikes.py 
```

### MIMIC ICU dataset:
You need to have the MIMICIII database running on a server. Run the following scripts to query and preprocess the ICU mortality data (This step might take a few hours)
```
python3 data_generator/icu_mortality.py --sqluser YOUR_USER --sqlpass YOUR_PASSWORD
```
```
python3 data_generator/data_preprocess.py
```

## Running the importance assignment baselines
For running the experiments, you need to train: 1) The black-box predictor model and 2) the conditional generator. Run the folowing script to train the models for your required dataset (state_simulation, spike_simulation, mimic)
```
python3 ./train.sh DATASET_NAME
```
Once you have the trained models, run the experiment string to generate feature importance results for all baselines
```
python3 -m TSX.main --data DATASET_NAME
```
