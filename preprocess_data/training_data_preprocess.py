from re import L
import argparse

import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import KFold
import pdb

speaker_id_arr_dict = {'msp-improv': np.arange(0, 12, 1), 
                       'crema-d': np.arange(1001, 1092, 1),
                       'iemocap': np.arange(0, 10, 1)}

if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='mel_spec')
    parser.add_argument('--input_channel', default=1)
    parser.add_argument('--input_spec_size', default=128)
    parser.add_argument('--aug', default='')
    parser.add_argument('--norm', default='min_max')
    parser.add_argument('--win_len', default=200)
    parser.add_argument('--validate', default=1)
    parser.add_argument('--shift', default=1)
    args = parser.parse_args()

    data_set_str, feature_type = args.dataset, args.feature_type
    shift = 'shift' if int(args.shift) == 1 else 'without_shift'
    win_len, feature_len = int(args.win_len), int(args.input_spec_size)
    aug = '_aug_emotion'if args.aug == 'emotion' else '_aug_gender'
    
    root_path = Path('/media/data/projects/speech-privacy')
    
    # get the cross validation sets
    speaker_id_arr = speaker_id_arr_dict[data_set_str]    
    train_array, test_array, validate_array = [], [], []
    kf = KFold(n_splits=5, random_state=8, shuffle=True) if data_set_str == 'crema-d' else KFold(n_splits=5, random_state=None, shuffle=False)
    
    for train_index, test_index in kf.split(speaker_id_arr):
        tmp_arr = speaker_id_arr[train_index]
        if int(args.validate) == 1:
            tmp_validate_len = int(len(tmp_arr) * 0.25)
            tmp_train_arr = tmp_arr[tmp_validate_len:]
            tmp_validate_arr = tmp_arr[:tmp_validate_len]
            validate_array.append(tmp_validate_arr)
        train_array.append(tmp_train_arr)
        test_array.append(speaker_id_arr[test_index])

    # if we dont have data ready for experiments, preprocess them first
    for i in range(5):
        
        test_fold = 'fold' + str(int(i)+1)
        preprocess_path = root_path.joinpath('preprocessed_data', shift, feature_type, str(feature_len))
        tmp_path = preprocess_path.joinpath(data_set_str, test_fold, 'training_'+str(win_len)+'_'+args.norm+aug+'.pkl')
            
        if os.path.exists(tmp_path) is False:
            cmd_str = 'python3 preprocess_data.py --dataset ' + data_set_str
            cmd_str += ' --test_fold ' + 'fold' + str(i+1)
            cmd_str += ' --feature_type ' + feature_type
            cmd_str += ' --feature_len ' + str(feature_len)
            if args.aug is not '':
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