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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import math
from copy import deepcopy

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import EarlyStopping, SpeechDataGenerator
from training_tools import speech_collate, setup_seed, seed_worker, get_class_weight
from baseline_models import one_d_cnn_lstm, two_d_cnn_lstm, two_d_cnn, deep_two_d_cnn_lstm, deep_two_d_cnn_lstm_tmp
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

def test(model, device, data_loader, optimizer, loss, epoch, args, pred='emotion'):
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
            preds = model(tmp_features, global_feature=global_data) if int(args.global_feature) == 1 else model(tmp_features)
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
    
    tmp_result_dict = {}
    tmp_result_dict[args.dataset] = {}
    tmp_result_dict[args.dataset]['acc'] = {}
    tmp_result_dict[args.dataset]['rec'] = {}
    tmp_result_dict[args.dataset]['loss'] = {}
    tmp_result_dict[args.dataset]['conf'] = {}
    
    acc_score = accuracy_score(truth_dict[args.dataset], predict_dict[args.dataset])
    rec_score = recall_score(truth_dict[args.dataset], predict_dict[args.dataset], average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(truth_dict[args.dataset], predict_dict[args.dataset], normalize='true')*100, decimals=2)

    print('Total test accuracy %.3f / recall %.3f after {%d}' % (acc_score, rec_score, epoch))
    print(confusion_matrix_arr)

    tmp_result_dict[args.dataset]['acc'][args.pred] = acc_score
    tmp_result_dict[args.dataset]['rec'][args.pred] = rec_score
    tmp_result_dict[args.dataset]['conf'][args.pred] = confusion_matrix_arr

    if 'combine' in args.dataset:
        for tmp_str in tmp_list:
            tmp_result_dict[tmp_str] = {}
            tmp_result_dict[tmp_str]['acc'] = {}
            tmp_result_dict[tmp_str]['rec'] = {}
            tmp_result_dict[tmp_str]['loss'] = {}
            tmp_result_dict[tmp_str]['conf'] = {}

            acc_score = accuracy_score(truth_dict[tmp_str], predict_dict[tmp_str])
            rec_score = recall_score(truth_dict[tmp_str], predict_dict[tmp_str], average='macro')
            confusion_matrix_arr = np.round(confusion_matrix(truth_dict[tmp_str], predict_dict[tmp_str], normalize='true')*100, decimals=2)

            print('%s: total test accuracy %.3f / recall %.3f after {%d}' % (tmp_str, acc_score, rec_score, epoch))
            print(confusion_matrix_arr)

            tmp_result_dict[tmp_str]['acc'][args.pred] = acc_score
            tmp_result_dict[tmp_str]['rec'][args.pred] = rec_score
            tmp_result_dict[tmp_str]['conf'][args.pred] = confusion_matrix_arr
    
    return tmp_result_dict


def train(model, device, data_loader, optimizer, loss, epoch, args, mode='training', pred='emotion'):
    if mode == 'training':
        model.train()
    else:
        model.eval()

    train_loss_list, total_loss_list = [], []
    
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

        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[0]]))[:, :, :]
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[1]]))
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[2]]))
        lengths = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[3]])).squeeze()
        global_data = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[4]]))
        dataset_data = [dataset for dataset in sampled_batch[5]]
        speaker_id_data = [str(speaker_id) for speaker_id in sampled_batch[7]]
 
        features, labels_emo, labels_gen, lengths, global_data = features.to(device), labels_emo.to(device), labels_gen.to(device), lengths.to(device), global_data.to(device)
        if len(features.shape) == 3: features = features.unsqueeze(dim=1)
        features, labels_emo, labels_gen, lengths, global_data = Variable(features), Variable(labels_emo), Variable(labels_gen), Variable(lengths), Variable(global_data)
        
        labels_arr = labels_emo if pred == 'emotion' else labels_gen
        preds = model(features, global_feature=global_data) if int(args.global_feature) == 1 else model(features)

        # calculate loss
        if 'combine' in args.dataset:
            total_loss = 0
            for pred_idx in range(len(preds)):
                speaker_id = speaker_id_data[pred_idx]+'_'+dataset_data[pred_idx]
                total_loss += loss(preds[pred_idx].unsqueeze(dim=0), labels_arr[pred_idx]) * weights[speaker_id]
            total_loss = total_loss / len(preds)
        total_loss_list.append(total_loss.item())
        train_loss_list.append(total_loss.item())

        # step the loss back
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
    if args.optimizer == 'adam':
        mean_loss = np.mean(train_loss_list)
        if mode == 'validate':
            scheduler.step(mean_loss)
            print('validate loss step')
    else:
        scheduler.step()

    tmp_result_dict = {}
    tmp_result_dict[args.dataset] = {}
    tmp_result_dict[args.dataset]['acc'] = {}
    tmp_result_dict[args.dataset]['rec'] = {}
    tmp_result_dict[args.dataset]['loss'] = {}
    tmp_result_dict[args.dataset]['conf'] = {}
    
    acc_score = accuracy_score(truth_dict[args.dataset], predict_dict[args.dataset])
    rec_score = recall_score(truth_dict[args.dataset], predict_dict[args.dataset], average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(truth_dict[args.dataset], predict_dict[args.dataset], normalize='true')*100, decimals=2)
    mean_loss = np.mean(train_loss_list)
    
    print('Total %s accuracy %.3f / recall %.3f / loss %.3f after {%d}' % (mode, acc_score, rec_score, mean_loss, epoch))
    print(confusion_matrix_arr)

    tmp_result_dict[args.dataset]['acc'][args.pred] = acc_score
    tmp_result_dict[args.dataset]['rec'][args.pred] = rec_score
    tmp_result_dict[args.dataset]['loss'][args.pred] = mean_loss
    tmp_result_dict[args.dataset]['conf'][args.pred] = confusion_matrix_arr

    if args.dataset == 'combine':
        for tmp_str in ['iemocap', 'crema-d', 'msp-improv']:
            tmp_result_dict[tmp_str] = {}
            tmp_result_dict[tmp_str]['acc'] = {}
            tmp_result_dict[tmp_str]['rec'] = {}
            tmp_result_dict[tmp_str]['loss'] = {}
            tmp_result_dict[tmp_str]['conf'] = {}

            acc_score = accuracy_score(truth_dict[tmp_str], predict_dict[tmp_str])
            rec_score = recall_score(truth_dict[tmp_str], predict_dict[tmp_str], average='macro')
            confusion_matrix_arr = np.round(confusion_matrix(truth_dict[tmp_str], predict_dict[tmp_str], normalize='true')*100, decimals=2)

            print('%s: total %s accuracy %.3f / recall %.3f after {%d}' % (tmp_str, mode, acc_score, rec_score, epoch))
            print(confusion_matrix_arr)

            tmp_result_dict[tmp_str]['acc'][args.pred] = acc_score
            tmp_result_dict[tmp_str]['rec'][args.pred] = rec_score
            tmp_result_dict[tmp_str]['conf'][args.pred] = confusion_matrix_arr
    
    return tmp_result_dict
            

if __name__ == '__main__':

    torch.cuda.empty_cache() 
    torch.multiprocessing.set_sharing_strategy('file_system')

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--input_spec_size', default=64)
    parser.add_argument('--cnn_filter_size', type=int, default=32)
    parser.add_argument('--num_emo_classes', default=4)
    parser.add_argument('--num_gender_class', default=2)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=50)
    parser.add_argument('--model_type', default='cnn-lstm-att')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--global_feature', default=0)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--shift', default=1)
    parser.add_argument('--att', default=None)
    parser.add_argument('--adv', default=0)

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
    
    # we want to do 5 validation
    save_result_df = pd.DataFrame()
    for config_type in model_parameters_dict:
        
        feature_len = model_parameters_dict[config_type]['feature_len']
        hidden_size = model_parameters_dict[config_type]['hidden']
        filter_size = model_parameters_dict[config_type]['filter']
        att_size = model_parameters_dict[config_type]['att_size']
        
        for i in range(5):
            torch.cuda.empty_cache()

            save_row_str = 'fold'+str(int(i+1))
            row_df = pd.DataFrame(index=[save_row_str])
            
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

            if 'combine' in args.dataset:
                # tmp_list = ['iemocap', 'crema-d', 'msp-improv'] if args.dataset == 'combine' else ['iemocap', 'crema-d']
                weights = {}
                for key in train_dict:
                    speaker_id = str(train_dict[key]['speaker_id'])+'_'+train_dict[key]['dataset']
                    if speaker_id not in weights:
                        weights[speaker_id] = 0
                    weights[speaker_id] += 1
                
                for key in validate_dict:
                    speaker_id = str(validate_dict[key]['speaker_id'])+'_'+validate_dict[key]['dataset']
                    if speaker_id not in weights:
                        weights[speaker_id] = 0
                    weights[speaker_id] += 1
            
                weights = get_class_weight(weights)
                print(weights)
            
            # Data loaders
            dataset_train = SpeechDataGenerator(train_dict, list(train_dict.keys()), mode='train', input_channel=int(args.input_channel))
            dataloader_train = DataLoader(dataset_train, worker_init_fn=seed_worker, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=speech_collate)
            
            dataset_val = SpeechDataGenerator(validate_dict, list(validate_dict), mode='validation', input_channel=int(args.input_channel))
            dataloader_val = DataLoader(dataset_val, worker_init_fn=seed_worker, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=speech_collate)

            dataset_test = SpeechDataGenerator(test_dict, list(test_dict), input_channel=int(args.input_channel))
            dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False, collate_fn=speech_collate)

            # Model related
            device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available(): print('GPU available, use GPU')

            if args.model_type == '1d-cnn-lstm':
                model = one_d_cnn_lstm(input_channel=int(args.input_channel), 
                                       input_spec_size=feature_len, 
                                       cnn_filter_size=filter_size, 
                                       pred=args.pred,
                                       lstm_hidden_size=hidden_size, 
                                       num_layers_lstm=2, 
                                       attention_size=att_size,
                                       att=args.att,
                                       global_feature=int(args.global_feature))
            elif args.model_type == '2d-cnn':
                model = two_d_cnn(input_channel=int(args.input_channel), 
                                  input_spec_size=feature_len, 
                                  cnn_filter_size=filter_size, 
                                  pred=args.pred,
                                  global_feature=int(args.global_feature))
            elif args.model_type == 'deep-2d-cnn-lstm':
                model = deep_two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                            input_spec_size=feature_len, 
                                            cnn_filter_size=filter_size, 
                                            pred=args.pred,
                                            lstm_hidden_size=hidden_size, 
                                            num_layers_lstm=2, 
                                            attention_size=att_size,
                                            att=args.att,
                                            global_feature=int(args.global_feature))
            elif args.model_type == 'tmp':
                model = deep_two_d_cnn_lstm_tmp(input_channel=int(args.input_channel), 
                                                input_spec_size=feature_len, 
                                                cnn_filter_size=filter_size, 
                                                pred=args.pred,
                                                lstm_hidden_size=hidden_size, 
                                                num_layers_lstm=2, 
                                                attention_size=att_size,
                                                att=args.att,
                                                global_feature=int(args.global_feature))
            else:
                model = two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                       input_spec_size=feature_len, 
                                       cnn_filter_size=filter_size, 
                                       pred=args.pred,
                                       lstm_hidden_size=hidden_size, 
                                       num_layers_lstm=2, 
                                       attention_size=att_size,
                                       att=args.att,
                                       global_feature=int(args.global_feature))

            model = model.to(device)
            loss = nn.CrossEntropyLoss().to(device)
            
            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=10, verbose=True)

            # initialize the optimizer
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) 
            elif args.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2, verbose=True)

            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print(params)

            best_val_recall, final_recall, best_epoch, final_confusion = 0, 0, 0, 0
            best_val_acc, final_acc = 0, 0
            result_dict = {}
            
            num_epochs = 100 if args.optimizer == 'sgd' else args.num_epochs
            for epoch in range(num_epochs):
                
                # perform the training, validate, and test
                train_result = train(model, device, dataloader_train, optimizer, loss, epoch, args, mode='training', pred=args.pred)
                validate_result = train(model, device, dataloader_val, optimizer, loss, epoch, args, mode='validate', pred=args.pred)
                test_result = test(model, device, dataloader_test, optimizer, loss, epoch, args, pred=args.pred)
                
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
                    best_model = deepcopy(model.state_dict())

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                if epoch > 10:
                    early_stopping(validate_result[args.dataset]['loss'][args.pred], model)

                row_df['config'] = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size)
                row_df['acc'] = final_acc
                row_df['rec'] = final_recall
                row_df['epoch'] = best_epoch
                # print(final_acc, best_val_acc, best_epoch)
                print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, final_acc*100, best_val_acc*100))
                print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, final_recall*100, best_val_recall*100))
                print('hidden size %d, filter size: %d, att size: %d' % (hidden_size, filter_size, att_size))
                print(test_result[args.dataset]['conf'][args.pred])
                
                if args.optimizer != 'sgd':
                    if early_stopping.early_stop and epoch > 10:
                        print("Early stopping")
                        break
            
            root_result_str = '2022_icassp_result'
            save_result_df = pd.concat([save_result_df, row_df])
            save_global_feature = 'with_global' if int(args.global_feature) == 1 else 'without_global'
            save_aug = 'aug_'+args.norm+'_'+str(int(args.win_len))+'_'+args.norm
            model_param_str = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size) if args.att is not None else 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size)
            # so if it is trained using adv dataset or service provider dataset
            exp_result_str = 'baseline_result' if int(args.adv) == 0 else 'adv_baseline_result'
            
            model_result_path = Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, args.pred, save_row_str)
            
            create_folder(Path.cwd().parents[0].joinpath(root_result_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, args.pred))
            create_folder(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, model_param_str, args.pred, save_row_str))
            
            torch.save(best_model, str(model_result_path.joinpath('model.pt')))

            f = open(str(model_result_path.joinpath('results_'+str(args.input_spec_size)+'.pkl')), "wb")
            pickle.dump(result_dict, f)
            f.close()

            save_result_df.to_csv(str(Path.cwd().parents[0].joinpath(root_result_str, exp_result_str, save_global_feature, save_aug, args.model_type, args.feature_type, args.dataset, args.input_spec_size, 'result_'+args.input_spec_size+'_'+args.pred+'.csv')))
    


