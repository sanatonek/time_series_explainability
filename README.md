# Instance-wise Feature Importance in Time (FIT)

FIT is a framework for explaining time series perdiction models, by assigning feature importance to every observation over time. [paper](https://papers.nips.cc/paper/2020/hash/08fa43588c2571ade19bc0fa5936e028-Abstract.html)

To run the experiments, you need a trained prediction model that takes in time series data as input, and generates a prediction over time. You also need the training data to train the FIT generator. Below are the instruction for replicating experiments in the paper.

## Data preparation
Two different simulated datasets are used in the experiments. The process of creating the data is explained below.


### Simulated dataset (State data):
Run the following script to create the data and the ground thruth explanations for the state experiment. You can choose the total number of samples in the dataset as well as the lenght of each recording. The defaults are set to 1000 samples of length 100.
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
Run the following scripts to query and preprocess the ICU mortality data (This step might take a few hours)
```
python3 data_generator/icu_mortality.py ---sqluser YOUR_USER --sqlpass YOUR_PASSWORD
```

## Running the importance assignment baselines
For running the experiments, you need to train: 1) The black-box predictor model and 2) the conditional generator. You can do this by passing the --train argument. If a model and conditional generator is already trained, skip the '--train' argument. To generate explanations for test samples using any of the baselines and for your required dataset (simulation, simulation_spike, mimic), run the following module.

```
python3 -m evaluation.baselines --data DATASET_NAME --explainer EXPLAINER_MODEL --train
```
In addition to FIT, you can also run experiments on different baseline explainers such as retain, deep lift, feature occlusion, etc.
