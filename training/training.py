from re import L
import torch
from torch.utils.data import DataLoader
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

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import EarlyStopping, SpeechDataGenerator
from training_tools import speech_collate, setup_seed, seed_worker
from baseline_models import one_d_cnn_lstm, two_d_cnn_lstm
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

    predict_list, truth_list = [], []
    predict_emotion_list, predict_gender_list = [], []
    truth_emotion_list, truth_gender_list = [], []
    
    for batch_idx, sampled_batch in enumerate(data_loader):
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[0]]))
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[1]]))
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[2]]))
        global_data = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[4]]))
        global_data = global_data.to(device)

        feature_len = features.shape[2] if features.shape[1] == 3 else features.shape[2]
        
        test_len = int((feature_len - win_len) / shift_len) + 1
        pred_list, pred_emotion_list, pred_gender_list = [], [], []
        
        for test_idx in range(test_len):
            tmp_features = features[0, :, test_idx*shift_len:test_idx*shift_len+win_len, :].to(device)
            tmp_features = tmp_features.unsqueeze(dim=0)
            
            labels_arr = labels_emo if pred == 'emotion' else labels_gen
        
            preds = model(tmp_features, global_feature=global_data) if int(args.global_feature) == 1 else model(tmp_features)
            m = nn.Softmax(dim=1)

            if args.pred == 'multitask':
                preds_emotion = m(preds[0])
                preds_gender = m(preds[1])

                pred_emotion_list.append(preds_emotion.detach().cpu().numpy()[0])
                pred_gender_list.append(preds_gender.detach().cpu().numpy()[0])
            else:
                preds = m(preds)
                pred_list.append(preds.detach().cpu().numpy()[0])
        
        # pdb.set_trace()
        if args.pred == 'multitask':
            mean_emotion_predictions = np.mean(np.array(pred_emotion_list), axis=0)
            mean_gender_predictions = np.mean(np.array(pred_gender_list), axis=0) 
            
            predict_emotion_list.append(np.argmax(mean_emotion_predictions))
            predict_gender_list.append(np.argmax(mean_gender_predictions))

            truth_emotion_list.append(labels_emo.detach().cpu().numpy()[0][0])
            truth_gender_list.append(labels_gen.detach().cpu().numpy()[0][0])
        else:
            mean_predictions = np.mean(np.array(pred_list), axis=0)
            prediction = np.argmax(mean_predictions)

            predict_list.append(prediction)
            truth_list.append(labels_arr.detach().cpu().numpy()[0][0])
    
    tmp_result_dict = {}
    tmp_result_dict['acc'] = {}
    tmp_result_dict['rec'] = {}
    tmp_result_dict['loss'] = {}
    tmp_result_dict['conf'] = {}
    
    # pdb.set_trace()
    if args.pred == 'multitask':

        emotion_rec_score = recall_score(truth_emotion_list, predict_emotion_list, average='macro')
        emotion_acc_score = accuracy_score(truth_emotion_list, predict_emotion_list)
        gender_rec_score = recall_score(truth_gender_list, predict_gender_list, average='macro')
        gender_acc_score = accuracy_score(truth_gender_list, predict_gender_list)

        print('Total test emotion accuracy %.3f / recall %.3f after {%d}' % (emotion_acc_score, emotion_rec_score, epoch))
        print('Total test gender accuracy %.3f / recall %.3f after {%d}' % (gender_acc_score, gender_rec_score, epoch))

        confusion_emotion_matrix_arr = confusion_matrix(truth_emotion_list, predict_emotion_list, normalize='true')
        confusion_emotion_matrix_arr = np.round(confusion_emotion_matrix_arr, decimals=4)
        print(confusion_emotion_matrix_arr*100)

        confusion_gender_matrix_arr = confusion_matrix(truth_gender_list, predict_gender_list, normalize='true')
        confusion_gender_matrix_arr = np.round(confusion_gender_matrix_arr, decimals=4)
        print(confusion_gender_matrix_arr*100)
        
        tmp_result_dict['acc']['emotion'] = emotion_acc_score
        tmp_result_dict['acc']['gender'] = gender_acc_score
        tmp_result_dict['rec']['emotion'] = emotion_rec_score
        tmp_result_dict['rec']['gender'] = gender_rec_score
        tmp_result_dict['conf']['emotion'] = confusion_emotion_matrix_arr
        tmp_result_dict['conf']['gender'] = confusion_gender_matrix_arr
    else:
        acc_score = accuracy_score(truth_list, predict_list)
        rec_score = recall_score(truth_list, predict_list, average='macro')
        confusion_matrix_arr = confusion_matrix(truth_list, predict_list, normalize='true')
        confusion_matrix_arr = np.round(confusion_matrix_arr, decimals=4)

        print('Total test accuracy %.3f / recall %.3f after {%d}' % (acc_score, rec_score, epoch))
        print(confusion_matrix_arr*100)

        tmp_result_dict['acc'][args.pred] = rec_score
        tmp_result_dict['rec'][args.pred] = acc_score
        tmp_result_dict['conf'][args.pred] = confusion_matrix_arr
    
    return tmp_result_dict


def train(model, device, data_loader, optimizer, loss, epoch, args, mode='training', pred='emotion'):
    if mode == 'training':
        model.train()  # set training mode
    else:
        model.eval()

    train_loss_list, total_loss_list = [], []
    emotion_loss_list, gender_loss_list = [], []

    predict_list, truth_list = [], []
    predict_emotion_list, predict_gender_list = [], []
    truth_emotion_list, truth_gender_list = [], []
    

    for batch_idx, sampled_batch in enumerate(data_loader):

        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[0]]))[:, :, :]
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[1]]))
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[2]]))
        lengths = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[3]])).squeeze()
        global_data = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[4]]))
 
        features, labels_emo, labels_gen, lengths, global_data = features.to(device), labels_emo.to(device), labels_gen.to(device), lengths.to(device), global_data.to(device)
        if len(features.shape) == 3:
            features = features.unsqueeze(dim=1)
        features, labels_emo, labels_gen, lengths, global_data = Variable(features), Variable(labels_emo), Variable(labels_gen), Variable(lengths), Variable(global_data)
        
        labels_arr = labels_emo if pred == 'emotion' else labels_gen
        preds = model(features, global_feature=global_data) if int(args.global_feature) == 1 else model(features)

        if args.pred == 'multitask':
            total_loss1 = loss(preds[0], labels_emo[0]) if len(labels_emo) == 1 else loss(preds[0], labels_emo.squeeze())
            total_loss2 = loss(preds[1], labels_gen[0]) if len(labels_gen) == 1 else loss(preds[1], labels_gen.squeeze())
            total_loss = 0.75*total_loss1 + 0.25*total_loss2

            emotion_loss_list.append(total_loss1.item())
            gender_loss_list.append(total_loss2.item())
        else:
            total_loss = loss(preds, labels_arr[0]) if len(labels_emo) == 1 else loss(preds, labels_arr.squeeze())
        total_loss_list.append(total_loss.item())

        if mode == 'training':
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        train_loss_list.append(total_loss.item())
        if args.pred == 'multitask':
            predictions_emotion = np.argmax(preds[0].detach().cpu().numpy(), axis=1)
            predictions_gender = np.argmax(preds[1].detach().cpu().numpy(), axis=1)

            for pred_idx in range(len(predictions_emotion)):
                predict_emotion_list.append(predictions_emotion[pred_idx])
                predict_gender_list.append(predictions_gender[pred_idx])

                truth_emotion_list.append(labels_emo.detach().cpu().numpy()[pred_idx][0])
                truth_gender_list.append(labels_gen.detach().cpu().numpy()[pred_idx][0])
        else:
            predictions = np.argmax(preds.detach().cpu().numpy(), axis=1)
            for pred_idx in range(len(predictions)):
                predict_list.append(predictions[pred_idx])
                truth_list.append(labels_arr.detach().cpu().numpy()[pred_idx][0])
            
        if batch_idx % 20 == 0:
            print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)), batch_idx))

    # pdb.set_trace()

    if args.optimizer == 'adam':
        if mode == 'validate':
            scheduler.step(total_loss)
            print('validate loss step')
    else:
        scheduler.step()

    tmp_result_dict = {}
    tmp_result_dict['acc'] = {}
    tmp_result_dict['rec'] = {}
    tmp_result_dict['loss'] = {}
    
    print('learnign rate %f', optimizer.param_groups[0]['lr'])
    if args.pred == 'multitask':

        emotion_rec_score = recall_score(truth_emotion_list, predict_emotion_list, average='macro')
        emotion_acc_score = accuracy_score(truth_emotion_list, predict_emotion_list)
        gender_rec_score = recall_score(truth_gender_list, predict_gender_list, average='macro')
        gender_acc_score = accuracy_score(truth_gender_list, predict_gender_list)

        mean_emotion_loss = np.mean(emotion_loss_list)
        mean_gender_loss = np.mean(gender_loss_list)

        if mode == 'training':
            print('Total training emotion accuracy %.3f / recall %.3f / loss %.3f after {%d}' % (emotion_acc_score, emotion_rec_score, mean_emotion_loss, epoch))
            print('Total training gender accuracy %.3f / recall %.3f / loss %.3f after {%d}' % (gender_acc_score, gender_rec_score, mean_gender_loss, epoch))
        else:
            print('Total validation emotion accuracy %.3f / recall %.3f / loss %.3f after {%d}' % (emotion_acc_score, emotion_rec_score, mean_emotion_loss, epoch))
            print('Total validation gender accuracy %.3f / recall %.3f / loss %.3f after {%d}' % (gender_acc_score, gender_rec_score, mean_gender_loss, epoch))

        confusion_matrix_arr = confusion_matrix(truth_emotion_list, predict_emotion_list, normalize='true')
        confusion_matrix_arr = np.round(confusion_matrix_arr, decimals=4)
        print(confusion_matrix_arr*100)

        confusion_matrix_arr = confusion_matrix(truth_gender_list, predict_gender_list, normalize='true')
        confusion_matrix_arr = np.round(confusion_matrix_arr, decimals=4)
        print(confusion_matrix_arr*100)
        
        tmp_result_dict['acc']['emotion'] = emotion_acc_score
        tmp_result_dict['acc']['gender'] = gender_acc_score
        tmp_result_dict['rec']['emotion'] = emotion_rec_score
        tmp_result_dict['rec']['gender'] = gender_rec_score
        tmp_result_dict['loss']['emotion'] = mean_emotion_loss
        tmp_result_dict['loss']['gender'] = mean_gender_loss
    
    else:
        rec_score = recall_score(truth_list, predict_list, average='macro')
        acc_score = accuracy_score(truth_list, predict_list)
        mean_loss = np.mean(train_loss_list)

        if mode == 'training':
            print('Total training accuracy %.3f / recall %.3f / loss %.3f after {%d}' % (acc_score, rec_score, mean_loss, epoch))
        else:
            print('Total validation accuracy %.3f / recall %.3f / loss %.3f after {%d}' % (acc_score, rec_score, mean_loss, epoch))

        confusion_matrix_arr = confusion_matrix(truth_list, predict_list, normalize='true')
        confusion_matrix_arr = np.round(confusion_matrix_arr, decimals=4)
        print(confusion_matrix_arr*100)

        tmp_result_dict['acc'][args.pred] = rec_score
        tmp_result_dict['rec'][args.pred] = acc_score
        tmp_result_dict['loss'][args.pred] = mean_loss
    
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
    parser.add_argument('--batch_size', default=30)
    parser.add_argument('--aug', default=None)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=50)
    parser.add_argument('--model_type', default='cnn-lstm-att')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--global_feature', default=0)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--validate', default=1)
    parser.add_argument('--shift', default=1)
    parser.add_argument('--att', default=None)

    args = parser.parse_args()
    data_set_str, feature_type = args.dataset, args.feature_type
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    win_len = int(args.win_len)

    if args.aug is None:
        aug = ''
    else:
        aug = '_aug_emotion'if args.aug == 'emotion' else '_aug_gender'
    
    setup_seed(8)
    torch.manual_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    
    # generate training parameters
    model_parameters_dict = {}
    hidden_size_list = [128]
    filter_size_list = [64]
    att_size_list = [64] if 'global' in args.model_type else [64, 128]

    for feature_len in [int(args.input_spec_size)]:
        for hidden_size in hidden_size_list:
            for filter_size in filter_size_list:
                for att_size in att_size_list:
                    config_type = 'feature_len_' + str(feature_len) + '_hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size)
                    model_parameters_dict[config_type] = {}
                    model_parameters_dict[config_type]['feature_len'] = feature_len
                    model_parameters_dict[config_type]['hidden'] = hidden_size
                    model_parameters_dict[config_type]['filter'] = filter_size
                    model_parameters_dict[config_type]['att_size'] = att_size

    # get the cross validation sets
    speaker_id_arr = speaker_id_arr_dict[data_set_str]    
    train_array, test_array, validate_array = [], [], []
    kf = KFold(n_splits=5, random_state=8, shuffle=True) if data_set_str == 'crema-d' else KFold(n_splits=5, random_state=None, shuffle=False)
    
    for train_index, test_index in kf.split(speaker_id_arr):
        tmp_arr = speaker_id_arr[train_index]
        if int(args.validate) == 1:
            # tmp_train_arr, tmp_validate_arr = train_test_split(tmp_arr, test_size=int(np.round(len(tmp_arr)*0.1)), random_state=8)
            tmp_train_arr = tmp_arr[2:]
            tmp_validate_arr = tmp_arr[:2]
            validate_array.append(tmp_validate_arr)
        train_array.append(tmp_train_arr)
        test_array.append(speaker_id_arr[test_index])
    
    # if we dont have data ready for experiments, preprocess them first
    for i in range(5):
        if data_set_str == 'msp-podcast' and i != 0:
            continue

        test_fold = 'fold' + str(int(i)+1)
        preprocess_path = root_path.joinpath('preprocessed_data', shift, feature_type, str(feature_len))
        tmp_path = preprocess_path.joinpath(data_set_str, test_fold, 'training_'+str(win_len)+'_'+args.norm+aug+'.pkl')
            
        if os.path.exists(tmp_path) is False:
            cmd_str = 'python3 ../preprocess_data/preprocess_data.py --dataset ' + data_set_str
            cmd_str += ' --test_fold ' + 'fold' + str(i+1)
            cmd_str += ' --feature_type ' + feature_type
            cmd_str += ' --feature_len ' + str(feature_len)
            if args.aug is not None:
                cmd_str += ' --aug ' + args.aug
            cmd_str += ' --win_len ' + str(win_len)
            cmd_str += ' --norm ' + args.norm
            cmd_str += ' --shift ' + args.shift
            cmd_str += ' --train_arr '
            for train_idx in train_array[i]:
                cmd_str += str(train_idx) + ' '
            cmd_str += ' --test_arr '
            for test_idx in test_array[i]:
                cmd_str += str(test_idx) + ' '
            if int(args.validate) == 1:
                cmd_str += ' --validation_arr '
                for validate_idx in validate_array[i]:
                    cmd_str += str(validate_idx) + ' '
            print(cmd_str)
            os.system(cmd_str)
    
    # we want to do 5 validation
    save_result_df = pd.DataFrame()
    
    for config_type in model_parameters_dict:
        
        feature_len = model_parameters_dict[config_type]['feature_len']
        hidden_size = model_parameters_dict[config_type]['hidden']
        filter_size = model_parameters_dict[config_type]['filter']
        att_size = model_parameters_dict[config_type]['att_size']
        
        for i in range(5):
            torch.cuda.empty_cache()

            if data_set_str == 'msp-podcast' and i != 0:
                continue

            save_row_str = 'fold'+str(int(i+1))
            row_df = pd.DataFrame(index=[save_row_str])

            with open(preprocess_path.joinpath(data_set_str, save_row_str, 'training_'+str(win_len)+'_'+args.norm+aug+'.pkl'), 'rb') as f:
                train_dict = pickle.load(f)
            with open(preprocess_path.joinpath(data_set_str, save_row_str, 'validation_'+str(win_len)+'_'+args.norm+aug+'.pkl'), 'rb') as f:
                validate_dict = pickle.load(f)
            with open(preprocess_path.joinpath(data_set_str, save_row_str, 'test_'+str(win_len)+'_'+args.norm+aug+'.pkl'), 'rb') as f:
                test_dict = pickle.load(f)
            
            # Data loaders
            dataset_train = SpeechDataGenerator(train_dict, list(train_dict.keys()), mode='train', input_channel=int(args.input_channel))
            dataloader_train = DataLoader(dataset_train, worker_init_fn=seed_worker, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=speech_collate)
            
            if int(args.validate) == 1:
                dataset_val = SpeechDataGenerator(validate_dict, list(validate_dict), mode='validation', input_channel=int(args.input_channel))
                dataloader_val = DataLoader(dataset_val, worker_init_fn=seed_worker, batch_size=args.batch_size, num_workers=0, shuffle=True, collate_fn=speech_collate)

            dataset_test = SpeechDataGenerator(test_dict, list(test_dict), input_channel=int(args.input_channel))
            dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=0, shuffle=False, collate_fn=speech_collate)

            # Model related
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available(): print('GPU available, use GPU')

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
                optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 
            elif args.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print(params)

            best_val_recall, final_recall, best_epoch, final_confusion = 0, 0, 0, 0
            best_val_acc, final_acc = 0, 0
            result_dict = {}
            for epoch in range(args.num_epochs):
                
                # perform the training, validate, and test
                train_result = train(model, device, dataloader_train, optimizer, loss, epoch, args, mode='training', pred=args.pred)
                if int(args.validate) == 1:
                    validate_result = train(model, device, dataloader_val, optimizer, loss, epoch, args, mode='validate', pred=args.pred)
                test_result = test(model, device, dataloader_test, optimizer, loss, epoch, args, pred=args.pred)
                
                # save the results for later
                result_dict[epoch] = {}
                result_dict[epoch]['train'] = train_result
                result_dict[epoch]['test'] = test_result
                if int(args.validate) == 1:
                    result_dict[epoch]['validate'] = validate_result
                
                if args.pred == 'multitask':
                    if validate_result['acc']['emotion'] > best_val_acc:
                        best_val_acc = validate_result['acc']['emotion']
                    final_emotion_acc = test_result['acc']['emotion']
                    final_emotion_recall = test_result['rec']['emotion']
                    final_emotion_confusion = test_result['conf']['emotion']
                    # best_epoch = epoch

                    row_df['config'] = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size)
                    row_df['emotion_acc'] = test_result['acc']['emotion']
                    row_df['emotion_rec'] = test_result['rec']['emotion']
                    row_df['gender_acc'] = test_result['acc']['gender']
                    row_df['gender_rec'] = test_result['rec']['gender']
                    row_df['epoch'] = best_epoch
                    # print(final_acc, best_val_acc, best_epoch)
                    print('best epoch %d, best final emotion acc %.2f, best val acc %.2f' % (best_epoch, final_emotion_acc*100, best_val_acc*100))
                    print('best epoch %d, best final gender acc %.2f, best val acc %.2f' % (best_epoch, test_result['rec']['gender']*100, test_result['rec']['gender']*100))
                    print('hidden size %d, filter size: %d, att size: %d' % (hidden_size, filter_size, att_size))
                    print(test_result['conf']['emotion'])
                    print(test_result['conf']['gender'])
                else:
                    if validate_result['acc'][args.pred] > best_val_acc and epoch > 10:
                        best_val_acc = validate_result['acc'][args.pred]
                        best_val_recall = validate_result['rec'][args.pred]
                        final_acc = test_result['acc'][args.pred]
                        final_recall = test_result['rec'][args.pred]
                        final_confusion = test_result['conf'][args.pred]
                        best_epoch = epoch

                    row_df['config'] = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size)
                    row_df['acc'] = final_acc
                    row_df['rec'] = final_recall
                    row_df['epoch'] = best_epoch
                    # print(final_acc, best_val_acc, best_epoch)
                    print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, final_acc*100, best_val_acc*100))
                    print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, final_recall*100, best_val_recall*100))
                    print('hidden size %d, filter size: %d, att size: %d' % (hidden_size, filter_size, att_size))
                    print(test_result['conf'][args.pred])

                    # early_stopping needs the validation loss to check if it has decresed, 
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping(validate_result['loss'][args.pred], model)
                
                if early_stopping.early_stop and epoch > 10:
                    print("Early stopping")
                    break
            
            save_result_df = pd.concat([save_result_df, row_df])
            save_global_feature = 'with_global' if int(args.global_feature) == 1 else 'without_global'
            if args.aug is None:
                save_aug = 'without_aug_'+str(win_len)+'_'+args.norm
            else:
                save_aug = 'with_aug_emotion_'+str(win_len)+'_'+args.norm if args.aug == 'emotion' else 'with_aug_gender_'+str(win_len)+'_'+args.norm
            model_param_str = 'hidden_'+str(hidden_size) + 'filter_'+str(filter_size) + 'att_'+str(att_size) if args.att is not None else 'hidden_'+str(hidden_size) + 'filter_'+str(filter_size)
            
            model_result_path = Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred, save_row_str)
            
            create_folder(Path.cwd().parents[0].joinpath('model_result'))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len)))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred))
            create_folder(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred, save_row_str))
            
            torch.save(model.state_dict(), str(model_result_path.joinpath('model.pt')))

            f = open(str(model_result_path.joinpath('results_'+str(args.input_spec_size)+'.pkl')), "wb")
            pickle.dump(result_dict, f)
            f.close()

            save_result_df.to_csv(str(Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), 'result_'+str(args.input_spec_size)+'_'+args.pred+'.csv')))
    


