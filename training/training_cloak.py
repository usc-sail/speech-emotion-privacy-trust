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
import pdb
from sklearn.metrics import confusion_matrix

import sys, os
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'model'))
sys.path.append(os.path.join(os.path.abspath(os.path.curdir), '..', 'utils'))

from training_tools import EarlyStopping, SpeechDataGenerator
from training_tools import speech_collate, setup_seed, seed_worker
from baseline_models import one_d_cnn_lstm, two_d_cnn_lstm
from cloak_models import cloak_noise, two_d_cnn_lstm_syn
from torch.autograd import Variable

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

    predict_list, truth_list = [], []
    
    for batch_idx, sampled_batch in enumerate(data_loader):
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[0]]))
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[1]]))
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[2]]))
        global_data = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sampled_batch[4]]))
        global_data = global_data.to(device)

        feature_len = features.shape[2] if features.shape[1] == 3 else features.shape[2]
        
        test_len = int((feature_len - win_len) / shift_len) + 1
        pred_list = []
        
        for test_idx in range(test_len):
            tmp_features = features[0, :, test_idx*shift_len:test_idx*shift_len+win_len, :].to(device)
            tmp_features = tmp_features.unsqueeze(dim=0)
            
            labels_arr = labels_emo if pred == 'emotion' else labels_gen

            # if pred == 'gender':
            #    preds, noisy = model(tmp_features, global_feature=global_data, mask=mask) if int(args.global_feature) == 1 else model(tmp_features, mask=mask)
            #    preds = gender_model(noisy, global_feature=global_data) if int(args.global_feature) == 1 else gender_model(noisy)
            # else:
            preds, noisy = model(tmp_features, global_feature=global_data, mask=mask) if int(args.global_feature) == 1 else model(tmp_features, mask=mask)
            m = nn.Softmax(dim=1)

            preds = m(preds)
            pred_list.append(preds.detach().cpu().numpy()[0])
        
        mean_predictions = np.mean(np.array(pred_list), axis=0)
        prediction = np.argmax(mean_predictions)

        predict_list.append(prediction)
        truth_list.append(labels_arr.detach().cpu().numpy()[0][0])

    tmp_result_dict = {}
    tmp_result_dict['acc'] = {}
    tmp_result_dict['rec'] = {}
    tmp_result_dict['loss'] = {}
    tmp_result_dict['conf'] = {}
    
    acc_score = accuracy_score(truth_list, predict_list)
    rec_score = recall_score(truth_list, predict_list, average='macro')
    confusion_matrix_arr = confusion_matrix(truth_list, predict_list, normalize='true')
    confusion_matrix_arr = np.round(confusion_matrix_arr, decimals=4)

    print('Total test accuracy %.3f / recall %.3f after {%d}' % (acc_score, rec_score, epoch))
    print(confusion_matrix_arr*100)

    tmp_result_dict['acc'][args.pred] = acc_score
    tmp_result_dict['rec'][args.pred] = rec_score
    tmp_result_dict['conf'][args.pred] = confusion_matrix_arr
    
    return tmp_result_dict


def train(model, device, data_loader, optimizer, loss, epoch, args, mode='training', pred='emotion', mask=None):
    if mode == 'training':
        model.train()  # set training mode
    else:
        model.eval()

    train_loss_list, total_loss_list = [], []
    predict_list, truth_list = [], []
    
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
        preds, noisy = model(features, global_feature=global_data, mask=mask) if int(args.global_feature) == 1 else model(features, mask=mask)

        total_loss = loss(preds, labels_arr[0]) if len(labels_emo) == 1 else loss(preds, labels_arr.squeeze())
        total_loss_list.append(total_loss.item())

        if mode == 'training':
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        train_loss_list.append(total_loss.item())
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

    tmp_result_dict['acc'][args.pred] = acc_score
    tmp_result_dict['rec'][args.pred] = rec_score
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
    parser.add_argument('--suppression_ratio', default=0)

    args = parser.parse_args()
    data_set_str, feature_type = args.dataset, args.feature_type
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    win_len = int(args.win_len)
    aug = '_aug_'+args.aug
    
    setup_seed(8)
    torch.manual_seed(8)

    root_path = Path('/media/data/projects/speech-privacy')
    
    # generate training parameters
    model_parameters_dict = {}
    hidden_size_list = [128]
    filter_size_list = [64]
    att_size_list = [64] if 'global' in args.model_type else [64, 128]

    preprocess_path = root_path.joinpath('preprocessed_data', shift, feature_type, str(int(args.input_spec_size)))
        
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

            mus = torch.zeros((1, int(win_len), feature_len)).to(device)
            scale = torch.ones((1, int(win_len), feature_len)).to(device)

            noise_model = cloak_noise(mus, scale, torch.tensor(0.01).to(device), torch.tensor(5).to(device), device)
            noise_model = noise_model.to(device)

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

                gender_model = two_d_cnn_lstm(input_channel=int(args.input_channel), 
                                              input_spec_size=feature_len, 
                                              cnn_filter_size=filter_size, 
                                              pred='gender',
                                              lstm_hidden_size=hidden_size, 
                                              num_layers_lstm=2, 
                                              attention_size=att_size,
                                              att=args.att,
                                              global_feature=int(args.global_feature))
            
            save_global_feature = 'with_global' if int(args.global_feature) == 1 else 'without_global'
            save_aug = 'with_aug_'+args.aug+'_'+str(win_len)+'_'+args.norm
            model_param_str = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size) if args.att is not None else 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size)

            model_result_path = Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred, save_row_str)
            gender_model_result_path = Path.cwd().parents[0].joinpath('model_result', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, 'gender', save_row_str)
            
            loss = nn.CrossEntropyLoss().to(device)

            model = model.to(device)
            model.load_state_dict(torch.load(str(model_result_path.joinpath('model.pt'))))

            gender_model = gender_model.to(device)
            gender_model.load_state_dict(torch.load(str(gender_model_result_path.joinpath('model.pt'))))
            
            cloak_model = two_d_cnn_lstm_syn(model.to(device), noise_model.to(device))
            cloak_model = cloak_model.to(device)

            if int(args.suppression_ratio) != 0:
                cloak_model_result_path = Path.cwd().parents[0].joinpath('cloak_result', 'suppression_0', save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred, save_row_str)
                cloak_model.load_state_dict(torch.load(str(cloak_model_result_path.joinpath('model.pt'))))

                tmp = np.nanpercentile(cloak_model.intermed.scales().detach().cpu().numpy(), int(args.suppression_ratio))
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
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, cloak_model.parameters()), lr=0.001, weight_decay=1e-04, betas=(0.9, 0.98), eps=1e-9)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.2, verbose=True)

            model_parameters = filter(lambda p: p.requires_grad, cloak_model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print(params)

            best_val_recall, final_recall, best_epoch, final_confusion = 0, 0, 0, 0
            best_val_acc, final_acc = 0, 0
            result_dict = {}
            for epoch in range(args.num_epochs):
                
                # perform the training, validate, and test
                train_result = train(cloak_model, device, dataloader_train, optimizer, loss, epoch, args, mode='training', pred=args.pred, mask=mask)
                if int(args.validate) == 1: validate_result = train(cloak_model, device, dataloader_val, optimizer, loss, epoch, args, mode='validate', pred=args.pred, mask=mask)
                test_result = test(cloak_model, device, dataloader_test, optimizer, loss, epoch, args, pred=args.pred, mask=mask)
                
                # save the results for later
                result_dict[epoch] = {}
                result_dict[epoch]['train'] = train_result
                result_dict[epoch]['test'] = test_result
                if int(args.validate) == 1:
                    result_dict[epoch]['validate'] = validate_result

                if validate_result['acc'][args.pred] > best_val_acc and epoch > 10:
                    best_val_acc = validate_result['acc'][args.pred]
                    best_val_recall = validate_result['rec'][args.pred]
                    final_acc = test_result['acc'][args.pred]
                    final_recall = test_result['rec'][args.pred]
                    final_confusion = test_result['conf'][args.pred]
                    best_epoch = epoch

                if epoch > 10:
                    early_stopping(validate_result['loss'][args.pred], model)

                row_df['config'] = 'hidden_'+str(hidden_size) + '_filter_'+str(filter_size) + '_att_'+str(att_size)
                row_df['acc'] = final_acc
                row_df['rec'] = final_recall
                row_df['epoch'] = best_epoch
                # print(final_acc, best_val_acc, best_epoch)
                print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, final_acc*100, best_val_acc*100))
                print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, final_recall*100, best_val_recall*100))
                print('hidden size %d, filter size: %d, att size: %d' % (hidden_size, filter_size, att_size))
                print(test_result['conf'][args.pred])
                
                print("mean scale %.2f" % torch.mean(cloak_model.intermed.scales().detach().cpu()))
                print("max scale %.2f" % torch.max(cloak_model.intermed.scales().detach().cpu()))
                print("min scale %.2f" % torch.min(cloak_model.intermed.scales().detach().cpu()))

                print("mean mu %.2f" % torch.mean(cloak_model.intermed.locs.detach().cpu()))
                print("max mu %.2f" % torch.max(cloak_model.intermed.locs.detach().cpu()))
                print("min mu %.2f" % torch.min(cloak_model.intermed.locs.detach().cpu()))

                print("mean noise %.2f" % torch.mean(cloak_model.intermed.sample_noise().detach().cpu()))
                print("max noise %.2f" % torch.max(cloak_model.intermed.sample_noise().detach().cpu()))
                print("min noise %.2f" % torch.min(cloak_model.intermed.sample_noise().detach().cpu()))
                
                print(cloak_model.intermed.scales().detach().cpu())
                print(cloak_model.intermed.locs.detach().cpu())

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                if early_stopping.early_stop and epoch > 10:
                    print("Early stopping")
                    break
            
            save_result_df = pd.concat([save_result_df, row_df])
            suppression_str = 'suppression_' + str(args.suppression_ratio)
            cloak_model_result_path = Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred, save_row_str)

            create_folder(Path.cwd().parents[0].joinpath('cloak_result'))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type, data_set_str))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len)))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred))
            create_folder(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), model_param_str, args.pred, save_row_str))
            
            torch.save(cloak_model.state_dict(), str(cloak_model_result_path.joinpath('model.pt')))

            f = open(str(cloak_model_result_path.joinpath('results_'+str(args.input_spec_size)+'.pkl')), "wb")
            pickle.dump(result_dict, f)
            f.close()

            save_result_df.to_csv(str(Path.cwd().parents[0].joinpath('cloak_result', suppression_str, save_global_feature, save_aug, args.model_type, feature_type, data_set_str, str(feature_len), 'result_'+str(args.input_spec_size)+'_'+args.pred+'.csv')))
    


