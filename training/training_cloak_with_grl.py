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
import math
from copy import deepcopy

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import EarlyStopping, SpeechDataGenerator, ReturnResultDict
from training_tools import speech_collate, setup_seed, seed_worker, get_class_weight
from baseline_models import one_d_cnn_lstm, two_d_cnn_lstm, deep_two_d_cnn_lstm
from cloak_models import cloak_noise, two_d_cnn_lstm_syn_with_grl
import pdb
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, KFold


emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
gender_dict = {'F': 0, 'M': 1}
speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

shift_len = 50

def create_folder(folder):
    if Path.exists(folder) is False:
        Path.mkdir(folder)

def test(model, device, data_loader, optimizer, loss, epoch, args, pred='emotion', mask=None):
    model.eval()
    predict_dict, truth_dict = {}, {}
    
    predict_dict[args.dataset] = []
    truth_dict[args.dataset] = []

    if args.dataset == 'combine':
        tmp_list = ['iemocap', 'crema-d', 'msp-improv']
    elif args.dataset == 'combine_two':
        tmp_list = ['iemocap', 'crema-d']

    if 'combine' in args.dataset:
        for tmp_str in tmp_list:
            predict_dict[tmp_str] = []
            truth_dict[tmp_str] = []
    
    for batch_idx, sampled_batch in enumerate(data_loader):

        # read all relavant data
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[0]]))
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[1]]))
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[2]]))
        global_data = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[4]]))
        global_data = global_data.to(device)
        dataset_data = [dataset for dataset in sampled_batch[5]]

        feature_len = features.shape[2] if features.shape[1] == 3 else features.shape[2]
        test_len = int((feature_len - int(args.win_len)) / shift_len) + 1
        pred_list = []
        
        for test_idx in range(test_len):
            tmp_features = features[0, :, test_idx*shift_len:test_idx*shift_len+int(args.win_len), :].to(device)
            tmp_features = tmp_features.unsqueeze(dim=0)
            
            labels_arr = labels_emo if pred == 'emotion' else labels_gen
            pooling = None if 'deep' in args.model_type else 'mean'
            preds, preds_grl, noisy = model(tmp_features, global_feature=global_data, mask=mask, grl=False, pooling=pooling) if int(args.global_feature) == 1 else model(tmp_features, mask=mask, grl=False, pooling=pooling)        
            m = nn.Softmax(dim=1)

            preds = m(preds)
            pred_list.append(preds.detach().cpu().numpy()[0])
        
        mean_predictions = np.mean(np.array(pred_list), axis=0)
        prediction = np.argmax(mean_predictions)

        if 'combine' in args.dataset:
            predict_dict[dataset_data[0]].append(prediction)
            truth_dict[dataset_data[0]].append(labels_arr.detach().cpu().numpy()[0][0])
        predict_dict[args.dataset].append(prediction)
        truth_dict[args.dataset].append(labels_arr.detach().cpu().numpy()[0][0])
    
    tmp_result_dict = ReturnResultDict(truth_dict, predict_dict, args.dataset, args.pred, mode='test', loss=None, epoch=epoch)
    return tmp_result_dict


def train(model, device, data_loader, optimizer, loss, epoch, args, mode='training', pred='emotion', mask=None):
    if mode == 'training':
        model.train()
    else:
        model.eval()

    train_loss_list = []
    
    # define the result dict
    predict_dict, truth_dict = {}, {}
    predict_dict[args.dataset] = []
    truth_dict[args.dataset] = []

    if args.dataset == 'combine':
        tmp_list = ['iemocap', 'crema-d', 'msp-improv']
    elif args.dataset == 'combine_two':
        tmp_list = ['iemocap', 'crema-d']

    if 'combine' in args.dataset:
        for tmp_str in tmp_list:
            predict_dict[tmp_str] = []
            truth_dict[tmp_str] = []
    
    for batch_idx, sampled_batch in enumerate(data_loader):
        
        # read all relavant data
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[0]]))[:, :, :]
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[1]]))
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[2]]))
        global_data = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[4]]))
        dataset_data = [dataset for dataset in sampled_batch[5]]
        speaker_id_data = [str(speaker_id) for speaker_id in sampled_batch[7]]
 
        features, labels_emo, labels_gen, global_data = features.to(device), labels_emo.to(device), labels_gen.to(device), global_data.to(device)
        if len(features.shape) == 3: features = features.unsqueeze(dim=1)
        features, labels_emo, labels_gen, global_data = Variable(features), Variable(labels_emo), Variable(labels_gen), Variable(global_data)
        
        labels_arr = labels_emo if pred == 'emotion' else labels_gen
        pooling = None if 'deep' in args.model_type else 'mean'
        preds, preds_grl, noisy = model(features, global_feature=global_data, mask=mask, grl=False, pooling=pooling) if int(args.global_feature) == 1 else model(features, mask=mask, grl=False, pooling=pooling)

        # calculate loss
        if 'combine' in args.dataset:
            total_loss = 0
            for pred_idx in range(len(preds)):
                # the weights are designed for imbalance number of samples between dataset
                # the gradient backpropagation from gender model side will be timed by -1
                # so the gender model can be trained normally but the addition noise will be invarient to gender
                speaker_id = speaker_id_data[pred_idx]+'_'+dataset_data[pred_idx]

                if mode == 'training':
                    total_loss += (loss(preds[pred_idx].unsqueeze(dim=0), labels_arr[pred_idx]) * weights[speaker_id]) / len(preds)
                    total_loss += (float(args.gender_lambda)*loss(preds_grl[pred_idx].unsqueeze(dim=0), labels_gen[pred_idx]) * weights[speaker_id]) / len(preds)
                else:
                    total_loss += (loss(preds[pred_idx].unsqueeze(dim=0), labels_arr[pred_idx])) / len(preds)
                    total_loss += (float(args.gender_lambda)*loss(preds_grl[pred_idx].unsqueeze(dim=0), labels_gen[pred_idx])) / len(preds)

            # if we are training sigma and mu at the same time, we can add the loss to the sigma term
            # otherwise we add the suppression, the sigma is freezed, and only mu will be trained
            if int(args.suppression_ratio) == 0:
                scale_loss = torch.log(torch.mean(cloak_model.intermed.scales()))
                total_loss = total_loss - float(args.scale_lamda)*scale_loss
            else:
                total_loss = total_loss
        train_loss_list.append(total_loss.item())

        # back propgation
        if mode == 'training':
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # get the prediction results
        predictions = np.argmax(preds.detach().cpu().numpy(), axis=1)
        if 'combine' in args.dataset:
            for pred_idx in range(len(predictions)):
                predict_dict[dataset_data[pred_idx]].append(predictions[pred_idx])
                truth_dict[dataset_data[pred_idx]].append(labels_arr.detach().cpu().numpy()[pred_idx][0])
            
        for pred_idx in range(len(predictions)):
            predict_dict[args.dataset].append(predictions[pred_idx])
            truth_dict[args.dataset].append(labels_arr.detach().cpu().numpy()[pred_idx][0])
            
        if batch_idx % 20 == 0:
            print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)), batch_idx))
        
    # if validate mode, step the loss
    if mode == 'validate':
        mean_loss = np.mean(train_loss_list)
        if args.optimizer == 'adam':
            scheduler.step(mean_loss)
        else:
            scheduler.step()

    tmp_result_dict = ReturnResultDict(truth_dict, predict_dict, args.dataset, args.pred, mode=mode, loss=np.mean(train_loss_list), epoch=epoch)
    return tmp_result_dict


if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--cnn_filter_size', type=int, default=32)
    parser.add_argument('--num_emo_classes', default=4)
    parser.add_argument('--num_gender_class', default=2)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=30)
    parser.add_argument('--model_type', default='cnn-lstm-att')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--global_feature', default=0)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--shift', default=1)
    parser.add_argument('--att', default=None)
    parser.add_argument('--adv', default=0)
    parser.add_argument('--suppression_ratio', default=0)
    parser.add_argument('--scale_lamda', default=0)
    parser.add_argument('--grl_lambda', default=0.1)
    parser.add_argument('--gender_lambda', default=0.1)
    
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
    exp_result_str = 'baseline_result' if int(args.adv) == 0 else 'adv_baseline_result'
    
    # we want to do 5 validation
    save_result_df = pd.DataFrame()
    for config_type in model_parameters_dict:
        
        feature_len = model_parameters_dict[config_type]['feature_len']
        hidden_size = model_parameters_dict[config_type]['hidden']
        filter_size = model_parameters_dict[config_type]['filter']
        att_size = model_parameters_dict[config_type]['att_size']
        
        for i in range(0, 5):
            torch.cuda.empty_cache()

            save_row_str = 'fold'+str(int(i+1))
            row_df = pd.DataFrame(index=[save_row_str])

            save_global_feature = 'with_global' if int(args.global_feature) == 1 else 'without_global'
            save_aug = 'aug_'+args.norm+'_'+str(int(args.win_len))+'_'+args.norm
            model_param_str = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size) if args.att is not None else 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size)
            suppression_str = 'suppression_' + str(args.suppression_ratio)
            root_result_str = '2022_icassp_result'
            if float(args.gender_lambda) == 0.1:
                scale_lamda_str = 'lamda_'+str(args.scale_lamda)+'_grl_'+str(args.grl_lambda)
            else:
                scale_lamda_str = 'lamda_'+str(args.scale_lamda)+'_grl_'+str(args.grl_lambda)+'_gender_'+str(args.gender_lambda)
            
            # we are training baseline models
            if int(args.adv) == 0:
                with open(preprocess_path.joinpath(args.dataset, save_row_str, 'training_'+str(int(args.win_len))+'_'+args.norm+'_aug_'+args.aug+'.pkl'), 'rb') as f:
                    train_dict = pickle.load(f)
                with open(preprocess_path.joinpath(args.dataset, save_row_str, 'validation_'+str(int(args.win_len))+'_'+args.norm+'_aug_'+args.aug+'.pkl'), 'rb') as f:
                    validate_dict = pickle.load(f)
            else:
                with open(preprocess_path.joinpath(args.dataset, save_row_str, 'adv_training_'+str(int(args.win_len))+'_'+args.norm+'_aug_'+args.aug+'.pkl'), 'rb') as f:
                    train_dict = pickle.load(f)
                with open(preprocess_path.joinpath(args.dataset, save_row_str, 'adv_validation_'+str(int(args.win_len))+'_'+args.norm+'_aug_'+args.aug+'.pkl'), 'rb') as f:
                    validate_dict = pickle.load(f)
            with open(preprocess_path.joinpath(args.dataset, save_row_str, 'test_'+str(int(args.win_len))+'_'+args.norm+'_aug_'+args.aug+'.pkl'), 'rb') as f:
                test_dict = pickle.load(f)

            '''
            if args.dataset == 'combine':
                weights = {}
                for tmp_str in ['iemocap', 'crema-d', 'msp-improv']:
                    weights[tmp_str] = 0
                for key in train_dict:
                    weights[train_dict[key]['dataset']] += 1
                weights = get_class_weight(weights)
            '''
            
            if 'combine' in args.dataset:
                weights = {}
                for key in train_dict:
                    speaker_id = str(train_dict[key]['speaker_id'])+'_'+train_dict[key]['dataset']
                    if speaker_id not in weights:
                        weights[speaker_id] = 0
                    weights[speaker_id] += 1
                weights = get_class_weight(weights)
            
            # Data loaders
            dataset_train = SpeechDataGenerator(train_dict, list(train_dict.keys()), mode='train', input_channel=int(args.input_channel))
            dataloader_train = DataLoader(dataset_train, worker_init_fn=seed_worker, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=speech_collate)
            
            dataset_val = SpeechDataGenerator(validate_dict, list(validate_dict), mode='validation', input_channel=int(args.input_channel))
            dataloader_val = DataLoader(dataset_val, worker_init_fn=seed_worker, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=speech_collate)

            dataset_test = SpeechDataGenerator(test_dict, list(test_dict), input_channel=int(args.input_channel))
            dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False, collate_fn=speech_collate)

            # Model related
            device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available(): print('GPU available, use GPU')

            # pdb.set_trace()

            # noise model
            mus = torch.zeros((1, int(args.win_len), feature_len)).to(device)
            scale = torch.ones((1, int(args.win_len), feature_len)).to(device)
            noise_model = cloak_noise(mus, scale, torch.tensor(0.01).to(device), torch.tensor(10).to(device), device)
            noise_model = noise_model.to(device)

            loss = nn.CrossEntropyLoss().to(device)

            if args.model_type == '1d-cnn-lstm-att':
                model = one_d_cnn_lstm(input_channel=int(args.input_channel), 
                                       input_spec_size=feature_len, 
                                       cnn_filter_size=filter_size, 
                                       pred=args.pred,
                                       lstm_hidden_size=hidden_size, 
                                       num_layers_lstm=2, 
                                       attention_size=att_size,
                                       att=args.att,
                                       global_feature=int(args.global_feature))
            elif args.model_type == 'deep-2d-cnn-lstm':
                pre_trained_baseline_model = deep_two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                                                 input_spec_size=feature_len, 
                                                                 cnn_filter_size=filter_size, 
                                                                 pred=args.pred,
                                                                 lstm_hidden_size=hidden_size, 
                                                                 num_layers_lstm=2, 
                                                                 attention_size=att_size,
                                                                 att=args.att,
                                                                 global_feature=int(args.global_feature))
                gender_model = deep_two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                                   input_spec_size=feature_len, 
                                                   cnn_filter_size=filter_size, 
                                                   pred=args.pred,
                                                   lstm_hidden_size=hidden_size, 
                                                   num_layers_lstm=2, 
                                                   attention_size=att_size,
                                                   att=args.att,
                                                   global_feature=int(args.global_feature))
            else:
                pre_trained_baseline_model = two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                                            input_spec_size=feature_len, 
                                                            cnn_filter_size=filter_size, 
                                                            pred=args.pred,
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

            # map to device
            pre_trained_baseline_model = pre_trained_baseline_model.to(device)
            gender_model = gender_model.to(device)

            # load the pretrained model
            baseline_model_result_path = Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, args.pred, save_row_str)
            pre_trained_baseline_model.load_state_dict(torch.load(str(baseline_model_result_path.joinpath('model.pt')), map_location=device))
            
            # load cloak models
            cloak_model = two_d_cnn_lstm_syn_with_grl(pre_trained_baseline_model.to(device), gender_model.to(device), noise_model.to(device), float(args.grl_lambda))
            cloak_model = cloak_model.to(device)
            
            if int(args.suppression_ratio) != 0:
                cloak_model_result_path = Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, 'suppression_0', save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, str(feature_len), model_param_str, args.pred, save_row_str)
                cloak_model.load_state_dict(torch.load(str(cloak_model_result_path.joinpath('model.pt')), map_location=device))
                cloak_model.intermed.rhos.requires_grad = False

                # mask the std that is above 100-suppresion_ratio percentile
                tmp = np.nanpercentile(cloak_model.intermed.scales().detach().cpu().numpy(), 100-int(args.suppression_ratio))
                mask = torch.where(cloak_model.intermed.scales()>tmp, torch.zeros(cloak_model.intermed.scales().shape).to(device), torch.ones(cloak_model.intermed.scales().shape).to(device))
            else:
                mask = None

            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=10, verbose=True)

            # initialize the optimizer
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, cloak_model.parameters()), lr=0.001, momentum=0.9, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 
            elif args.optimizer == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, cloak_model.parameters()), lr=0.0005, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

            model_parameters = filter(lambda p: p.requires_grad, cloak_model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print(params)

            best_val_recall, final_recall, best_epoch, final_confusion = 0, 0, 0, 0
            best_val_acc, final_acc = 0, 0
            result_dict = {}
            for epoch in range(args.num_epochs):
                
                # perform the training, validate, and test
                train_result = train(cloak_model, device, dataloader_train, optimizer, loss, epoch, args, mode='training', pred=args.pred, mask=mask)
                validate_result = train(cloak_model, device, dataloader_val, optimizer, loss, epoch, args, mode='validate', pred=args.pred, mask=mask)
                test_result = test(cloak_model, device, dataloader_test, optimizer, loss, epoch, args, pred=args.pred, mask=mask)
                
                # save the results for later
                result_dict[epoch] = {}
                result_dict[epoch]['train'] = train_result
                result_dict[epoch]['test'] = test_result
                result_dict[epoch]['validate'] = validate_result
                
                if validate_result[args.dataset]['acc'][args.pred] > best_val_acc and epoch > 10:
                    best_val_acc = validate_result[args.dataset]['acc'][args.pred]
                    best_val_recall = validate_result[args.dataset]['rec'][args.pred]
                    final_acc = test_result[args.dataset]['acc'][args.pred]
                    final_recall = test_result[args.dataset]['rec'][args.pred]
                    final_confusion = test_result[args.dataset]['conf'][args.pred]
                    best_epoch = epoch
                    
                    best_model = deepcopy(cloak_model.state_dict())

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                if epoch > 10:
                    early_stopping(validate_result[args.dataset]['loss'][args.pred], cloak_model)

                row_df['config'] = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size)
                row_df['acc'] = final_acc
                row_df['rec'] = final_recall
                row_df['epoch'] = best_epoch
                # print(final_acc, best_val_acc, best_epoch)
                print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, final_acc*100, best_val_acc*100))
                print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, final_recall*100, best_val_recall*100))
                print('hidden size %d, filter size: %d, att size: %d' % (hidden_size, filter_size, att_size))
                print(test_result[args.dataset]['conf'][args.pred])
                
                tmp_scales = cloak_model.intermed.scales().detach().cpu()
                tmp_mus = cloak_model.intermed.locs.detach().cpu()
                tmp_noises = cloak_model.intermed.sample_noise().detach().cpu()

                print("mean, max, min scale %.2f, %.2f, %.2f" % (torch.mean(tmp_scales), torch.max(tmp_scales), torch.min(tmp_scales)))
                print("mean, max, min mu %.2f, %.2f, %.2f" % (torch.mean(tmp_mus), torch.max(tmp_mus), torch.min(tmp_mus)))
                print("mean, max, min noise %.2f, %.2f, %.2f" % (torch.mean(tmp_noises), torch.max(tmp_noises), torch.min(tmp_noises)))
                
                print(cloak_model.intermed.scales().detach().cpu())
                print(cloak_model.intermed.locs.detach().cpu())
                
                if early_stopping.early_stop and epoch > 10:
                    print("Early stopping")
                    break
            
            save_result_df = pd.concat([save_result_df, row_df])
            cloak_model_result_path = Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, args.pred, save_row_str)
            
            create_folder(Path.cwd().parents[0].joinpath(root_result_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, args.pred))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, args.pred, save_row_str))
            
            torch.save(best_model, str(cloak_model_result_path.joinpath('model.pt')))

            f = open(str(cloak_model_result_path.joinpath('results_'+str(args.input_spec_size)+'.pkl')), "wb")
            pickle.dump(result_dict, f)
            f.close()

            save_result_df.to_csv(str(Path.cwd().parents[0].joinpath(root_result_str, 'cloak_grl_'+exp_result_str, scale_lamda_str, suppression_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, 'result_'+args.input_spec_size+'_'+args.pred+'.csv')))
    


