from re import L
import torch
from torch.utils.data import DataLoader, dataset
import torch.nn as nn
import argparse
from torch import optim
import torch.multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import torch
import pickle
from pathlib import Path
import pandas as pd

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import EarlyStopping, SpeechDataGenerator, ReturnResultDict
from training_tools import speech_collate, setup_seed, seed_worker
from baseline_models import one_d_cnn_lstm, two_d_cnn_lstm
from cloak_models import cloak_noise, two_d_cnn_lstm_syn, two_d_cnn_lstm_syn_with_grl
import pdb


emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
gender_dict = {'F': 0, 'M': 1}
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

shift_len = 50

def create_folder(folder):
    if Path.exists(folder) is False:
        Path.mkdir(folder)


def test(cloak_model, device, data_loader, args, mask=None):
    cloak_model.eval()
    baseline_model.eval()
    adversary_model.eval()

    predict_dict, truth_dict = {}, {}
    adv_predict_dict, adv_truth_dict = {}, {}
    
    predict_dict[args.dataset] = []
    truth_dict[args.dataset] = []
    adv_predict_dict[args.dataset] = []
    adv_truth_dict[args.dataset] = []

    if args.dataset == 'combine':
        for tmp_str in ['iemocap', 'crema-d', 'msp-improv']:
            predict_dict[tmp_str] = []
            truth_dict[tmp_str] = []
            adv_predict_dict[tmp_str] = []
            adv_truth_dict[tmp_str] = []
    
    for batch_idx, sampled_batch in enumerate(data_loader):
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[0]]))
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[1]]))
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[2]]))
        global_data = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[4]]))
        global_data = global_data.to(device)
        dataset_data = [dataset for dataset in sampled_batch[5]]

        feature_len = features.shape[2] if features.shape[1] == 3 else features.shape[2]
        test_len = int((feature_len - int(args.win_len)) / shift_len) + 1
        pred_list = []
        adv_pred_list = []
        
        for test_idx in range(test_len):
            tmp_features = features[0, :, test_idx*shift_len:test_idx*shift_len+int(args.win_len), :].to(device)
            tmp_features = tmp_features.unsqueeze(dim=0)
            
            # add noise to the data
            if int(args.grl) == 0:
                preds, noisy = cloak_model(tmp_features, global_feature=global_data, mask=mask) if int(args.global_feature) == 1 else cloak_model(tmp_features, mask=mask)
            else:
                preds, preds_grl, noisy = cloak_model(tmp_features, global_feature=global_data, mask=mask) if int(args.global_feature) == 1 else cloak_model(tmp_features, mask=mask)
            preds = baseline_model(noisy)
            adv_preds = adversary_model(noisy)
            
            m = nn.Softmax(dim=1)
            preds = m(preds)
            adv_preds = m(adv_preds)

            pred_list.append(preds.detach().cpu().numpy()[0])
            adv_pred_list.append(adv_preds.detach().cpu().numpy()[0])
        
        prediction = np.argmax(np.mean(np.array(pred_list), axis=0))
        adv_prediction = np.argmax(np.mean(np.array(adv_pred_list), axis=0))

        if args.dataset == 'combine':
            predict_dict[dataset_data[0]].append(prediction)
            truth_dict[dataset_data[0]].append(labels_emo.detach().cpu().numpy()[0][0])
            adv_predict_dict[dataset_data[0]].append(adv_prediction)
            adv_truth_dict[dataset_data[0]].append(labels_gen.detach().cpu().numpy()[0][0])

        predict_dict[args.dataset].append(prediction)
        truth_dict[args.dataset].append(labels_emo.detach().cpu().numpy()[0][0])
        adv_predict_dict[args.dataset].append(adv_prediction)
        adv_truth_dict[args.dataset].append(labels_gen.detach().cpu().numpy()[0][0])
    
    # get the result for gender prediction after we add noise
    baseline_result_dict = ReturnResultDict(truth_dict, predict_dict, args.dataset, 'emotion', epoch=0)
    adv_result_dict = ReturnResultDict(adv_truth_dict, adv_predict_dict, args.dataset, 'gender', epoch=0)

    return baseline_result_dict, adv_result_dict


if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--model_type', default='cnn-lstm-att')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--global_feature', default=0)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--shift', default=1)
    parser.add_argument('--att', default=None)
    parser.add_argument('--grl', default=0)
    parser.add_argument('--scale_lamda', default=0)

    args = parser.parse_args()
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    
    setup_seed(8)
    torch.manual_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    
    # generate training parameters
    model_parameters_dict = {}
    hidden_size_list = [64]
    filter_size_list = [64]
    att_size_list = [64] if 'global' in args.model_type else [128]

    for hidden_size in hidden_size_list:
        for filter_size in filter_size_list:
            for att_size in att_size_list:
                config_type = 'feature_len_' + args.input_spec_size + '_hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size)
                model_parameters_dict[config_type] = {}
                model_parameters_dict[config_type]['feature_len'] = int(args.input_spec_size)
                model_parameters_dict[config_type]['hidden'] = hidden_size
                model_parameters_dict[config_type]['filter'] = filter_size
                model_parameters_dict[config_type]['att_size'] = att_size

    preprocess_path = root_path.joinpath('2022_icassp', shift, args.feature_type, args.input_spec_size)
    
    # we want to do 5 validation, for now we only care about black box condition
    save_result_df = pd.DataFrame()

    result_dict = {}
    for suppression_ratio in [0, 20, 40, 60, 80]:
    # for suppression_ratio in [0, 40, 50]:
        result_dict[suppression_ratio] = {}
        for config_type in model_parameters_dict:
            
            feature_len = model_parameters_dict[config_type]['feature_len']
            hidden_size = model_parameters_dict[config_type]['hidden']
            filter_size = model_parameters_dict[config_type]['filter']
            att_size = model_parameters_dict[config_type]['att_size']
            
            suppression_ratio_dict = {}
            
            for fold_idx in range(5):
                torch.cuda.empty_cache()
                suppression_ratio_dict[fold_idx] = {}

                save_global_feature = 'with_global' if int(args.global_feature) == 1 else 'without_global'
                save_aug = 'aug_'+args.norm+'_'+str(int(args.win_len))+'_'+args.norm
                model_param_str = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size) if args.att is not None else 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size)
                suppression_str = 'suppression_' + str(suppression_ratio)
                root_result_str = '2022_icassp_result'
                scale_lamda_str = 'lamda_'+str(args.scale_lamda)
                
                # we only do test
                with open(preprocess_path.joinpath(args.dataset, 'fold'+str(fold_idx+1), 'test_'+str(int(args.win_len))+'_'+args.norm+'_aug_'+args.aug+'.pkl'), 'rb') as f:
                    test_dict = pickle.load(f)

                # Data loaders
                dataset_test = SpeechDataGenerator(test_dict, list(test_dict), input_channel=int(args.input_channel))
                dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False, collate_fn=speech_collate)

                # Model related
                device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
                if torch.cuda.is_available(): print('GPU available, use GPU')

                # noise model
                mus = torch.zeros((1, int(args.win_len), feature_len)).to(device)
                scale = torch.ones((1, int(args.win_len), feature_len)).to(device)
                noise_model = cloak_noise(mus, scale, torch.tensor(0.01).to(device), torch.tensor(5).to(device), device)
                noise_model = noise_model.to(device)

                baseline_model = two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                                input_spec_size=feature_len, 
                                                cnn_filter_size=filter_size, 
                                                pred='emotion',
                                                lstm_hidden_size=hidden_size, 
                                                num_layers_lstm=2, 
                                                attention_size=att_size,
                                                att=args.att,
                                                global_feature=int(args.global_feature))

                adversary_model = two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                                input_spec_size=feature_len, 
                                                cnn_filter_size=filter_size, 
                                                pred='gender',
                                                lstm_hidden_size=hidden_size, 
                                                num_layers_lstm=2, 
                                                attention_size=att_size,
                                                att=args.att,
                                                global_feature=int(args.global_feature))

                gender_model = two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                              input_spec_size=feature_len, 
                                              cnn_filter_size=filter_size, 
                                              pred='gender',
                                              lstm_hidden_size=hidden_size, 
                                              num_layers_lstm=2, 
                                              attention_size=att_size,
                                              att=args.att,
                                              global_feature=int(args.global_feature))

                baseline_model = baseline_model.to(device)
                adversary_model = adversary_model.to(device)
                gender_model = gender_model.to(device)
                
                # load cloak models
                cloak_model = two_d_cnn_lstm_syn(baseline_model.to(device), noise_model.to(device)) if int(args.grl) == 0 else two_d_cnn_lstm_syn_with_grl(baseline_model.to(device), gender_model.to(device), noise_model.to(device))
                cloak_model = cloak_model.to(device)

                # load cloak model with corresponding suppresion ratio
                if int(args.grl) == 0:
                    cloak_model_result_path = Path.cwd().parents[0].joinpath(root_result_str, 'cloak_baseline_result', scale_lamda_str, 'suppression_'+str(suppression_ratio), save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, str(feature_len), model_param_str, 'emotion', 'fold'+str(fold_idx+1))
                else:
                    cloak_model_result_path = Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_baseline_result', scale_lamda_str, 'suppression_'+str(suppression_ratio), save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, str(feature_len), model_param_str, 'emotion', 'fold'+str(fold_idx+1))
                cloak_model.load_state_dict(torch.load(str(cloak_model_result_path.joinpath('model.pt'))))
                cloak_model.intermed.rhos.requires_grad = False
                cloak_model.intermed.locs.requires_grad = False
                
                # load the original models
                model_result_path = Path.cwd().parents[0].joinpath(root_result_str, 'baseline_result', save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, 'emotion', 'fold'+str(fold_idx+1))
                baseline_model.load_state_dict(torch.load(str(model_result_path.joinpath('model.pt'))))

                model_result_path = Path.cwd().parents[0].joinpath(root_result_str, 'adv_baseline_result', save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, 'gender', 'fold'+str(fold_idx+1))  
                adversary_model.load_state_dict(torch.load(str(model_result_path.joinpath('model.pt'))))
       
                # suppresing mask locations
                if suppression_ratio == 0:
                    mask = None
                else:
                    tmp = np.nanpercentile(cloak_model.intermed.scales().detach().cpu().numpy(), int(suppression_ratio))
                    mask = torch.where(cloak_model.intermed.scales()>tmp, torch.zeros(cloak_model.intermed.scales().shape).to(device), torch.ones(cloak_model.intermed.scales().shape).to(device))
            
                # perform the test
                baseline_result_dict, adv_result_dict = test(cloak_model, device, dataloader_test, args, mask=mask)
                # test_result = test(adversary_model, cloak_model, device, dataloader_test, epoch, args, pred='gender', mask=mask)
                suppression_ratio_dict[fold_idx]['baseline'] = baseline_result_dict
                suppression_ratio_dict[fold_idx]['adv'] = adv_result_dict

                # pdb.set_trace()
            
            for tmp_str in ['combine', 'iemocap', 'crema-d', 'msp-improv']:
                baseline_acc_result_list, baseline_rec_result_list = [], []
                adv_acc_result_list, adv_rec_result_list = [], []
                for fold_idx in suppression_ratio_dict:
                    baseline_acc_result_list.append(suppression_ratio_dict[fold_idx]['baseline'][tmp_str]['acc']['emotion'])
                    baseline_rec_result_list.append(suppression_ratio_dict[fold_idx]['baseline'][tmp_str]['rec']['emotion'])
                    
                    adv_acc_result_list.append(suppression_ratio_dict[fold_idx]['adv'][tmp_str]['acc']['gender'])
                    adv_rec_result_list.append(suppression_ratio_dict[fold_idx]['adv'][tmp_str]['rec']['gender'])
                
                save_row_str = 'suppression_ratio_'+str(suppression_ratio)+'_'+tmp_str
                
                row_df = pd.DataFrame(index=[save_row_str])
                row_df['baseline_acc'] = np.mean(baseline_acc_result_list)
                row_df['baseline_rec'] = np.mean(baseline_rec_result_list)
                row_df['adv_acc'] = np.mean(adv_acc_result_list)
                row_df['adv_rec'] = np.mean(adv_rec_result_list)
                save_result_df = pd.concat([save_result_df, row_df])
            # pdb.set_trace()

    save_name = 'grl-' + str(args.scale_lamda) if int(args.grl) == 1 else 'non-grl-' + str(args.scale_lamda)
    save_result_df.to_csv(save_name+'.csv')
    # pdb.set_trace()
                
