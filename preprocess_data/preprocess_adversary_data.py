from pathlib import Path
from numpy.lib.npyio import save
import pandas as pd
import re
import pickle
import numpy as np
import argparse
import pdb
from collections import Counter
import torch
from torch.nn.modules.module import T
from tqdm import tqdm


def create_folder(folder):
    if Path.exists(folder) is False:
        Path.mkdir(folder)


def write_data_dict(tmp_dict, data, label, gender, speaker_id, padding):
    tmp_dict['label'] = label
    tmp_dict['gender'] = gender
    tmp_dict['speaker_id'] = speaker_id
    tmp_dict['global_data'] = global_data

    for data_idx in range(len(data)):
        training_norm_dict[speaker_id].append(data[data_idx, :])
    
    if padding == True:
        tmp_data = np.empty([win_len, data.shape[1]])
        tmp_data[:, :] = np.nan
        tmp_data[:len(data), :] = data
        tmp_df = pd.DataFrame(tmp_data)
        tmp_dict['data'] = np.array(tmp_df.fillna(0))
        tmp_data = np.array(tmp_df)
    else:
        tmp_dict['data'] = data
        tmp_data = tmp_dict['data']
    

def save_data_dict(save_data, data_stats_dict, label, gender, speaker_id):

    if int(args.shift) == 1:
        padding = True if len(save_data) < win_len else False
        save_len = 1 if len(save_data) < win_len else int((len(save_data) - win_len) / shift_len) + 1
    else:
        padding = True if len(save_data) < win_len else False
        save_len = 1

    # save for normalization later
    if speaker_id not in training_norm_dict:
        training_norm_dict[speaker_id] = []
        training_global_norm_dict[speaker_id] = []
    
    for i in range(save_len):
        if speaker_id in test_speaker_id_arr:
            data_stats_dict['test'][label] += 1
            test_dict[sentence_file+'_'+str(i)] = {}
            write_data_dict(test_dict[sentence_file+'_'+str(i)], save_data, label, gender, speaker_id, padding)
            break
        elif speaker_id in adv_validation_speaker_id_arr:
            data_stats_dict['adv_valid'][label] += 1
            adv_valid_dict[sentence_file+'_'+str(i)] = {}
            write_data_dict(adv_valid_dict[sentence_file+'_'+str(i)], save_data[i*shift_len:i*shift_len+win_len], label, gender, speaker_id, padding)
        elif speaker_id in adv_train_speaker_id_arr:
            data_stats_dict['adv_train'][label] += 1
            adv_training_dict[sentence_file+'_'+str(i)] = {}
            write_data_dict(adv_training_dict[sentence_file+'_'+str(i)], save_data[i*shift_len:i*shift_len+win_len], label, gender, speaker_id, padding)
        elif speaker_id in validation_speaker_id_arr:
            data_stats_dict['valid'][label] += 1
            valid_dict[sentence_file+'_'+str(i)] = {}
            write_data_dict(valid_dict[sentence_file+'_'+str(i)], save_data[i*shift_len:i*shift_len+win_len], label, gender, speaker_id, padding)
        elif speaker_id in train_speaker_id_arr:
            if args.aug is not None:
                if args.aug == 'emotion':
                    train_label_list.append(label)
                else:
                    train_label_list.append(gender)
            data_stats_dict['training'][label] += 1
            training_dict[sentence_file+'_'+str(i)] = {}
            write_data_dict(training_dict[sentence_file+'_'+str(i)], save_data[i*shift_len:i*shift_len+win_len], label, gender, speaker_id, padding)
            
    training_global_norm_dict[speaker_id].append(global_data)


def combine_data():

    data_dict = {}
    data_dict['training'] = {}
    data_dict['validation'] = {}
    data_dict['adv_training'] = {}
    data_dict['adv_validation'] = {}
    data_dict['test'] = {}

    for tmp_data_set_str in ['iemocap', 'crema-d', 'msp-improv']:
        setting_str_list = ['training', 'validation', 'adv_training', 'adv_validation', 'test']
        for setting_str in setting_str_list:
            with open(preprocess_path.joinpath(tmp_data_set_str, args.test_fold, setting_str+'_'+str(win_len)+'_'+args.norm+'_aug_'+args.aug+'.pkl'), 'rb') as f:
                tmp_dict = pickle.load(f)
                for key in tmp_dict:
                    data_dict[setting_str][key] = tmp_dict[key]
                    data_dict[setting_str][key]['dataset'] = tmp_data_set_str
    
    return data_dict['training'], data_dict['validation'], data_dict['adv_training'], data_dict['adv_validation'], data_dict['test']

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_len', default=128)
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--test_fold',  default='fold1')
    parser.add_argument('--aug', default=None)
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--test_id',  default=0)
    parser.add_argument('--shift',  default=1)
    parser.add_argument('--win_len',  default=400)
    parser.add_argument('--train_arr', nargs='*', type=int, default=None)
    parser.add_argument('--validation_arr', nargs='*', type=int, default=None)
    parser.add_argument('--adv_train_arr', nargs='*', type=int, default=None)
    parser.add_argument('--adv_validation_arr', nargs='*', type=int, default=None)
    parser.add_argument('--test_arr', nargs='*', type=int, default=None)
    
    args = parser.parse_args()

    # read args
    test_fold = args.test_fold
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    
    win_len, shift_len = int(args.win_len), int(int(args.win_len)/4)
    feature_len, feature_type = int(args.feature_len), args.feature_type
    
    train_arr, validation_arr = args.train_arr, args.validation_arr
    adv_train_arr, adv_validation_arr = args.adv_train_arr, args.adv_validation_arr
    test_arr = args.test_arr
    
    # save preprocess file
    root_path = Path('/media/data/projects/speech-privacy')
    create_folder(root_path.joinpath('2022_icassp'))
    create_folder(root_path.joinpath('2022_icassp', shift))
    create_folder(root_path.joinpath('2022_icassp', shift, feature_type))
    create_folder(root_path.joinpath('2022_icassp', shift, feature_type, str(feature_len)))
    preprocess_path = root_path.joinpath('2022_icassp', shift, feature_type, str(feature_len))

    # feature folder
    feature_path = root_path.joinpath('feature', feature_type)
    
    training_data_list, training_norm_len = [], []
    training_norm_dict = {}
    training_global_norm_dict = {}

    train_data_len_list = []

    for data_set_str in [args.dataset]:

        if data_set_str in ['iemocap', 'crema-d', 'msp-improv']:
            with open(feature_path.joinpath(data_set_str, 'data_'+str(feature_len)+'.pkl'), 'rb') as f:
                data_dict = pickle.load(f)
        
        training_dict, valid_dict, test_dict = {}, {}, {}
        adv_training_dict, adv_valid_dict = {}, {}
        data_stats_dict = {}

        for data_dict_str in ['training', 'valid', 'adv_train', 'adv_valid', 'test']:
            data_stats_dict[data_dict_str] = {}
            data_stats_dict[data_dict_str]['neu'] = 0
            data_stats_dict[data_dict_str]['ang'] = 0
            data_stats_dict[data_dict_str]['sad'] = 0
            data_stats_dict[data_dict_str]['hap'] = 0
        
        train_label_list = []
        
        if data_set_str == 'msp-podcast':
            
            # data root folder
            data_root_path = Path('/media/data').joinpath('sail-data')
            data_str = 'MSP-podcast'
            label_df = pd.read_csv(data_root_path.joinpath(data_str, 'Labels', 'labels_concensus.csv'), index_col=0)
            
            # pdb.set_trace()
            
            for recording_type in ['train', 'validate', 'test']:
                tmp_data_dict = data_dict[recording_type]

                # pdb.set_trace()
                sentence_file_list = list(tmp_data_dict.keys())
                sentence_file_list.sort()

                for sentence_file in sentence_file_list[:]:

                    if 'Test2' in label_df.loc[sentence_file, 'Split_Set']:
                        continue
                    
                    emotion = label_df.loc[sentence_file, 'EmoClass']
                    speaker_id = label_df.loc[sentence_file, 'SpkrID']
                    gender = label_df.loc[sentence_file, 'Gender'][0]
                    # pdb.set_trace()

                    if speaker_id == 'Unknown':
                        print('unknown id skip')
                        continue

                    speaker_len = len(label_df.loc[label_df['SpkrID'] == speaker_id])
                    if speaker_len < 10:
                        print('speaker sentence too short')
                        continue
                    
                    if emotion == 'N':
                        label = 'neu'
                    elif emotion == 'S':
                        label = 'sad'
                    elif emotion == 'H':
                        label = 'hap'
                    elif emotion == 'A':
                        label = 'ang'
                    
                    data = tmp_data_dict[sentence_file]
                    save_data = np.array(data['feature'])
                    session_id = recording_type
                    global_data = global_data_dict[recording_type][sentence_file]['feature']

                    training_session = 'train'
                    validation_session = 'validate'
                    test_session = 'test'
                    
                    print(validation_session, test_session.lower(), session_id)
                    save_data_dict(save_data, test_session.lower(), validation_session, data_stats_dict, label, gender, speaker_id)

        elif data_set_str == 'msp-improv':
            # data root folder
            sentence_file_list = list(data_dict.keys())
            sentence_file_list.sort()

            speaker_id_arr = ['M01', 'F01', 'M02', 'F02', 'M03', 'F03', 'M04', 'F04', 'M05', 'F05', 'M06', 'F06']

            train_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in train_arr]
            validation_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in validation_arr]
            adv_train_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in adv_train_arr]
            adv_validation_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in adv_validation_arr]
            test_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in test_arr]
            
            # data root folder
            data_root_path = Path('/media/data').joinpath('sail-data')
            data_str = 'MSP-IMPROV'
            
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_part = sentence_file.split('-')
                recording_type = sentence_part[-2][-1:]
                emotion = sentence_part[-4][-1:]
                gender = sentence_part[-3][:1]
                speaker_id = sentence_part[-3]

                # we keep improv data only
                if recording_type == 'P':
                    continue
                if recording_type == 'R':
                    continue

                if emotion == 'N':
                    label = 'neu'
                elif emotion == 'S':
                    label = 'sad'
                elif emotion == 'H':
                    label = 'hap'
                elif emotion == 'A':
                    label = 'ang'
                
                data = data_dict[sentence_file]
                save_data = np.array(data['mel1'])[0].T if feature_type == 'mel_spec' else np.array(data['mfcc'])[0][:40].T
                global_data = data['gemaps']
                
                save_data_dict(save_data, data_stats_dict, label, gender, speaker_id)

        elif data_set_str == 'crema-d':
            
            # speaker id for training, validation, and test
            train_speaker_id_arr = [tmp_idx for tmp_idx in train_arr]
            validation_speaker_id_arr = [tmp_idx for tmp_idx in validation_arr]
            adv_train_speaker_id_arr = [tmp_idx for tmp_idx in adv_train_arr]
            adv_validation_speaker_id_arr = [tmp_idx for tmp_idx in adv_validation_arr]
            test_speaker_id_arr = [tmp_idx for tmp_idx in test_arr]
            
            # data root folder
            data_root_path = Path('/media/data').joinpath('public-data')
            demo_df = pd.read_csv(str(data_root_path.joinpath(data_set_str, 'VideoDemographics.csv')), index_col=0)
            sentence_file_list = list(data_root_path.joinpath(data_set_str).glob('*.wav'))
           
            sentence_file_list.sort()
            speaker_id_arr = np.arange(1001, 1092, 1)
            
            for sentence_file in tqdm(sentence_file_list, ncols=100, miniters=100):
                sentence_file = str(sentence_file).split('/')[-1].split('.wav')[0]
                sentence_part = sentence_file.split('_')
                
                speaker_id = int(sentence_part[0])
                label = sentence_part[2].lower()
                
                if label == 'ang' or label == 'neu' or label == 'sad' or label == 'hap':
                    if sentence_file not in data_dict:
                        continue
                    data = data_dict[sentence_file]
                    global_data = data['gemaps']
                    save_data = np.array(data['mel1'])[0].T if feature_type == 'mel_spec' else np.array(data['mfcc'])[0][:40].T
                    session_id = int(sentence_part[0])
                    gender = 'M' if demo_df.loc[int(session_id), 'Sex'] == 'Male' else 'F'
                    speaker_id = int(sentence_file.split('_')[0])
                    save_data_dict(save_data, data_stats_dict, label, gender, speaker_id)

        elif data_set_str == 'iemocap':

            speaker_id_arr = ['Ses01F', 'Ses01M', 'Ses02F', 'Ses02M', 'Ses03F', 'Ses03M', 'Ses04F', 'Ses04M', 'Ses05F', 'Ses05M']
            
            # speaker id for training, validation, and test
            train_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in train_arr]
            validation_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in validation_arr]
            adv_train_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in adv_train_arr]
            adv_validation_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in adv_validation_arr]
            test_speaker_id_arr = [speaker_id_arr[tmp_idx] for tmp_idx in test_arr]
          
            # data root folder
            data_root_path = Path('/media/data').joinpath('sail-data')
            for session_id in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
                ground_truth_path_list = list(data_root_path.joinpath(data_set_str, session_id, 'dialog', 'EmoEvaluation').glob('*.txt'))
                for ground_truth_path in tqdm(ground_truth_path_list, ncols=100, miniters=100):
                    with open(str(ground_truth_path)) as f:
                        file_content = f.read()

                        useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
                        label_lines = re.findall(useful_regex, file_content)
                        for line in label_lines:
                            if 'Ses' in line:
                                sentence_file = line.split('\t')[-3]
                                label = line.split('\t')[-2]

                                if label == 'ang' or label == 'neu' or label == 'sad' or label == 'hap' or label == 'exc':
                                    if label == 'exc':
                                        label = 'hap'
                                    
                                    if 'impro' not in line:
                                        continue

                                    data = data_dict[sentence_file]
                                    global_data = data['gemaps']
                                    save_data = np.array(data['mel1'])[0].T if feature_type == 'mel_spec' else np.array(data['mfcc'])[0][:40].T
                                    gender = sentence_file.split('_')[-1][0]
                                    speaker_id = sentence_file.split('_')[0][:-1] + gender
                                    
                                    # print(speaker_id, gender, label)
                                    save_data_dict(save_data, data_stats_dict, label, gender, speaker_id)
        
        elif data_set_str == 'combine':
            training_dict, valid_dict, adv_training_dict, adv_valid_dict, test_dict = combine_data()
            
        # if we are not trying to combine the dataset, we should do the normalization or augmentation
        if data_set_str != 'combine':
            speaker_norm_dict, speaker_global_norm_dict = {}, {}
            for speaker_id in training_norm_dict:
                norm_data_list = training_norm_dict[speaker_id]
                speaker_norm_dict[speaker_id] = {}
                speaker_norm_dict[speaker_id]['mean'] = np.nanmean(np.array(norm_data_list).reshape(-1, feature_len), axis=0)
                speaker_norm_dict[speaker_id]['std'] = np.nanstd(np.array(norm_data_list).reshape(-1, feature_len), axis=0)
                speaker_norm_dict[speaker_id]['min'] = np.nanmin(np.array(norm_data_list).reshape(-1, feature_len), axis=0)
                speaker_norm_dict[speaker_id]['max'] = np.nanmax(np.array(norm_data_list).reshape(-1, feature_len), axis=0)

                norm_data_list = training_global_norm_dict[speaker_id]
                speaker_global_norm_dict[speaker_id] = {}
                speaker_global_norm_dict[speaker_id]['mean'] = np.nanmean(np.array(norm_data_list), axis=0)
                speaker_global_norm_dict[speaker_id]['std'] = np.nanstd(np.array(norm_data_list), axis=0)
                speaker_global_norm_dict[speaker_id]['min'] = np.nanmin(np.array(norm_data_list), axis=0)
                speaker_global_norm_dict[speaker_id]['max'] = np.nanmax(np.array(norm_data_list), axis=0)
            
            for tmp_dict in [training_dict, valid_dict, adv_training_dict, adv_valid_dict, test_dict]:
                for file_name in tmp_dict:
                    
                    speaker_id = tmp_dict[file_name]['speaker_id']
                    if args.norm == 'znorm':
                        tmp_data = (tmp_dict[file_name]['data'] - speaker_norm_dict[speaker_id]['mean']) / (speaker_norm_dict[speaker_id]['std']+1e-5)
                    elif args.norm == 'min_max':
                        tmp_data = (tmp_dict[file_name]['data'] - speaker_norm_dict[speaker_id]['min']) / (speaker_norm_dict[speaker_id]['max'] - speaker_norm_dict[speaker_id]['min'])
                        tmp_data = tmp_data * 2 - 1
                    
                    save_data = np.zeros((1, len(tmp_data), feature_len))
                    save_data[0] = tmp_data[:, :feature_len]
                    tmp_dict[file_name]['data'] = save_data

                    # global data
                    global_mean_array = speaker_global_norm_dict[speaker_id]['mean']
                    global_std_array = speaker_global_norm_dict[speaker_id]['std']                
                    tmp_dict[file_name]['global_data'] = (tmp_dict[file_name]['global_data']  - global_mean_array) / (global_std_array+1e-5)

            if args.aug is not None:
                tmp_list = []
                for label in Counter(train_label_list):
                    tmp_list.append(Counter(train_label_list)[label])
                max_label_size = np.max(tmp_list)

                for label in Counter(train_label_list):
                    label_count = Counter(train_label_list)[label]

                    if label_count == max_label_size:
                        continue
                    number_of_aug = max_label_size - label_count

                    aug_key_list = []
                    for key in training_dict:
                        if args.aug == 'emotion':
                            if training_dict[key]['label'] == label:
                                aug_key_list.append(key)
                        else:
                            if training_dict[key]['gender'] == label:
                                aug_key_list.append(key)
                            
                    aug_idx_list = np.random.randint(0, len(aug_key_list), size=number_of_aug)
                    for idx, aug_idx in enumerate(aug_idx_list):
                        key = aug_key_list[aug_idx]
                        tmp_data = training_dict[key]['data']

                        noise_to_add = torch.normal(0, 0.05, size=tmp_data.shape).numpy()
                        augmented_audio = tmp_data + noise_to_add

                        training_dict[key+'_'+str(idx)] = training_dict[key]
                        training_dict[key+'_'+str(idx)]['data'] = augmented_audio
        
        create_folder(preprocess_path.joinpath(data_set_str))
        create_folder(preprocess_path.joinpath(data_set_str, test_fold))
        aug = '_aug_' + args.aug
            
        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'training_'+str(win_len)+'_'+args.norm+aug+'.pkl')), "wb")
        pickle.dump(training_dict, f)
        f.close()

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'validation_'+str(win_len)+'_'+args.norm+aug+'.pkl')), "wb")
        pickle.dump(valid_dict, f)
        f.close()

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'adv_training_'+str(win_len)+'_'+args.norm+aug+'.pkl')), "wb")
        pickle.dump(adv_training_dict, f)
        f.close()

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'adv_validation_'+str(win_len)+'_'+args.norm+aug+'.pkl')), "wb")
        pickle.dump(adv_valid_dict, f)
        f.close()

        f = open(str(preprocess_path.joinpath(data_set_str, test_fold, 'test_'+str(win_len)+'_'+args.norm+aug+'.pkl')), "wb")
        pickle.dump(test_dict, f)
        f.close()
        
