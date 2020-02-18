#! /bin/bash
DATASET=$1
if [DATASET = 'simulation']
then
	python -u -m TSX.main --data simulation --model feature_generator_explainer --generator joint_RNN_generator --train --cv $cv
	python -u -m TSX.main --data simulation --model risk_predictor --train --cv $cv
	python -u -m TSX.main --data simulation --model risk_predictor --predictor attention --train --cv $cv
fi
if [DATASET = 'simulation_spike']
then
        python -u -m TSX.main --data simulation_spike --model feature_generator_explainer --generator joint_RNN_generator --train --cv $cv
        python -u -m TSX.main --data simulation_spike --model risk_predictor --train --cv $cv
        python -u -m TSX.main --data simulation_spike --model risk_predictor --predictor attention --train --cv $cv
fi
if [DATASET = 'mimic']
then
        python -u -m TSX.main --data mimic --model feature_generator_explainer --generator joint_RNN_generator --train --cv $cv
        python -u -m TSX.main --data mimic --model risk_predictor --train --cv $cv
        python -u -m TSX.main --data mimic --model risk_predictor --predictor attention --train --cv $cv
fi
