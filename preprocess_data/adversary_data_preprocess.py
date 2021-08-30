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
    
    if data_set_str != 'combine':
        # get the cross validation sets
        speaker_id_arr = speaker_id_arr_dict[data_set_str]    
        baseline_train_array, baseline_validate_array = [], []
        adversary_train_array, adversary_validate_array = [], []
        test_array = []

        kf = KFold(n_splits=5, random_state=8, shuffle=True) if data_set_str == 'crema-d' else KFold(n_splits=5, random_state=None, shuffle=False)
        
        for other_index, test_index in kf.split(speaker_id_arr):
            tmp_arr = speaker_id_arr[other_index]

            adversary_len = int(np.round(len(tmp_arr) * 0.5))

            # 40% are baseline, 40% are adversary, and 20% are test
            adversary_arr = tmp_arr[len(test_array):len(test_array)+adversary_len]
            baseline_arr = [tmp for tmp in tmp_arr if tmp not in adversary_arr]

            if int(args.validate) == 1:
                baseline_validate_len = int(np.round(len(baseline_arr) * 0.2))
                adversary_validate_len = int(np.round(len(baseline_arr) * 0.2))
                
                baseline_train_arr = baseline_arr[baseline_validate_len:]
                baseline_validate_arr = [tmp for tmp in baseline_arr if tmp not in baseline_train_arr]

                adversary_train_arr = adversary_arr[adversary_validate_len:]
                adversary_validate_arr = [tmp for tmp in adversary_arr if tmp not in adversary_train_arr]
                
            baseline_train_array.append(baseline_train_arr)
            baseline_validate_array.append(baseline_validate_arr)
            adversary_train_array.append(adversary_train_arr)
            adversary_validate_array.append(adversary_validate_arr)
            test_array.append(speaker_id_arr[test_index])

    # if we dont have data ready for experiments, preprocess them first
    for i in range(5):
        cmd_str = 'python3 preprocess_adversary_data.py --dataset ' + data_set_str
        cmd_str += ' --test_fold ' + 'fold' + str(i+1)
        cmd_str += ' --feature_type ' + feature_type
        cmd_str += ' --feature_len ' + str(feature_len)
        if args.aug is not '':
            cmd_str += ' --aug ' + args.aug
        cmd_str += ' --win_len ' + str(win_len)
        cmd_str += ' --norm ' + args.norm
        cmd_str += ' --shift ' + args.shift

        # we dont have these speaker array when combining dataset
        if data_set_str != 'combine':
            cmd_str += ' --train_arr '
            for train_idx in baseline_train_array[i]:
                cmd_str += str(train_idx) + ' '
            cmd_str += ' --validation_arr '
            for validate_idx in baseline_validate_array[i]:
                cmd_str += str(validate_idx) + ' '

            cmd_str += ' --adv_train_arr '
            for adv_train_idx in adversary_train_array[i]:
                cmd_str += str(adv_train_idx) + ' '
            cmd_str += ' --adv_validation_arr '
            for adv_validate_idx in adversary_validate_array[i]:
                cmd_str += str(adv_validate_idx) + ' '
                
            cmd_str += ' --test_arr '
            for test_idx in test_array[i]:
                cmd_str += str(test_idx) + ' '
        
        print(cmd_str)
        os.system(cmd_str)
