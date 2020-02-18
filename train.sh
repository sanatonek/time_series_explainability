#!/bin/bash
if [ $1 == 'simulation' ]
then
        python -u -m TSX.main --data simulation --model feature_generator_explainer --generator joint_RNN_generator --train
        python -u -m TSX.main --data simulation --model risk_predictor --train
        python -u -m TSX.main --data simulation --model risk_predictor --predictor attention --train
elif [ $1 == 'simulation_spike' ]
then
        python -u -m TSX.main --data simulation_spike --model feature_generator_explainer --generator joint_RNN_generator --train
        python -u -m TSX.main --data simulation_spike --model risk_predictor --train
        python -u -m TSX.main --data simulation_spike --model risk_predictor --predictor attention --train
elif [ $1 == 'mimic' ]
then
        python -u -m TSX.main --data mimic --model feature_generator_explainer --generator joint_RNN_generator --train
        python -u -m TSX.main --data mimic --model risk_predictor --train
        python -u -m TSX.main --data mimic --model risk_predictor --predictor attention --train
fi
