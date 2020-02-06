#!/bin/bash

for cv in 0 1 2
do
    python -u -m TSX.main --data mimic --model feature_generator_explainer --generator joint_RNN_generator --train --cv $cv
    python -u -m TSX.main --data mimic --model risk_predictor --train --cv $cv
    python -u -m TSX.main --data mimic --model risk_predictor --predictor attention --train --cv $cv
    python -u -m TSX.main --data mimic --model feature_generator_explainer --generator joint_RNN_generator --cv $cv --all_samples
done
