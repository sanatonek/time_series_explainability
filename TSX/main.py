import torch
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from TSX.utils import train_model, load_data, test, load_simulated_data, load_ghg_data
from TSX.models import DeepKnn
from TSX.experiments import Baseline, EncoderPredictor, FeatureGeneratorExplainer
import argparse
import numpy as np

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']
feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

def main(experiment, train, uncertainty_score, sensitivity=False, sim_data=False, data='mimic'):
    print('********** Experiment with the %s model **********' %(experiment))

    # Configurations
    encoding_size = 100
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs=120
    historical=True
    rnn_type='GRU'

    if sim_data:
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(batch_size=100, path='./data_generator/data/simulated_data')
        feature_size = p_data.shape[1]
        encoding_size=5
        if experiment =='feature_generator_explainer':
            n_epochs = 50
        elif experiment=='risk_predictor':
            n_epochs = 30
        data='simulation'
        historical=True
    else:
        if data=='mimic':
            p_data, train_loader, valid_loader, test_loader = load_data(batch_size)
            feature_size = p_data.feature_size
        elif data=='ghg':
            p_data, train_loader, valid_loader, test_loader = load_ghg_data(batch_size)
            feature_size = p_data.feature_size
            n_epochs=80
            rnn_type='GRU'
            encoding_size=100

    if experiment == 'baseline':
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == 'risk_predictor':
        exp = EncoderPredictor(train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type=rnn_type,simulation=sim_data,data=data)
    elif experiment == 'feature_generator_explainer':
        exp = FeatureGeneratorExplainer(train_loader, valid_loader, test_loader, feature_size, patient_data=p_data, generator_hidden_size=encoding_size, prediction_size=1, historical=historical, simulation=sim_data,data=data)

    # for id in range(len(intervention_list)):
        # exp.summary_stat(id)
        # exp.plot_summary_stat(id)
    # exp.plot_summary_stat(1)
    # for i in range(27):
    #     print('#### %s'%(feature_map_mimic[i]))
    #     print(torch.mean(exp.feature_dist_0[:,i,:]),torch.mean(exp.feature_dist_1[:,i,:]))
    exp.run(train=train,n_epochs=n_epochs)
    # span = []
    # # import matplotlib.pyplot as plt
    # testset = list(exp.test_loader.dataset)
    # # signals = torch.stack(([x[0] for x in testset]))
    # # plt.plot(np.array(signals[4126,2,:]))
    # # plt.show()
    # for i,(signal,label) in enumerate(testset):
    #     # if i==79:
    #     #     for j in range(31):
    #     #         plt.plot(signal[j,:].cpu().detach().numpy())
    #     #     plt.show()
    #     exp.risk_predictor.load_state_dict(torch.load('./ckpt/risk_predictor.pt'))
    #     exp.risk_predictor.to(device)
    #     exp.risk_predictor.eval()
    #     risk=[]
    #     for t in range(1,48):
    #         risk.append(exp.risk_predictor(signal[:, 0:t].view(1, signal.shape[0], t).to(device)).item())
    #     span.append((i,max(risk) - min(risk)))
    # span.sort(key= lambda pair:pair[1], reverse=True)
    # print([x[0] for x in span[0:300]])


    if sensitivity:
        #exp.model.train()
        for i, (signals, labels) in enumerate(train_loader):
            signals = torch.Tensor(signals.float()).to(device).requires_grad_()
            risks = exp.model(signals)
            risks[0].backward(retain_graph=True)
            print(signals.grad.data[0,:,:])


    if uncertainty_score:
        # Evaluate uncertainty using deep KNN method
        print('\n********** Uncertainty Evaluation: **********')
        sample_ind = 1
        n_nearest_neighbors = 10
        dknn = DeepKnn(exp.model, p_data.train_data[0:int(0.8 * p_data.n_train), :, :],
                    p_data.train_label[0:int(0.8 * p_data.n_train)], device)
        knn_labels = dknn.evaluate_confidence(sample=p_data.test_data[sample_ind, :, :].reshape((1, -1, 48)),
                                              sample_label=p_data.test_label[sample_ind],
                                              _nearest_neighbors=n_nearest_neighbors, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ICU mortality prediction model')
    parser.add_argument('--model', type=str, default='feature_generator_explainer', help='Prediction model')
    parser.add_argument('--simulation', action='store_true')
    parser.add_argument('--data', type=str, default='mimic')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    parser.add_argument('--sensitivity',action='store_true')
    args = parser.parse_args()
    main(args.model, train=args.train, uncertainty_score=args.uncertainty, sensitivity=args.sensitivity, sim_data=args.simulation, data=args.data)
