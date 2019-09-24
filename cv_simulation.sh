#!/bin/bash

for cv in 0 1 2 3 4
do
    python -u main.py --data simulation_spike --model feature_generator_explainer --generator joint_RNN_generator --train --cv $cv
    python -u main.py --data simulation_spike --model risk_predictor --train --cv $cv
    python -u main.py --data simulation_spike --model feature_generator_explainer --generator joint_RNN_generator --cv $cv
done

#for cv in 0 1 2 3 4
#do
#    python -u main.py --data simulation_spike --model feature_generator_explainer --generator joint_RNN_generator --train --cv $cv
#    python -u main.py --data simulation_spike --model risk_predictor --train --cv $cv
#    python -u main.py --data simulation_spike --model feature_generator_explainer --generator joint_RNN_generator --cv $cv
#done
