# This is the script for training data preprocess

# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
python3 training_data_preprocess.py --input_spec_size 128 --dataset iemocap \
                        --feature_type mel_spec --validate 1 --win_len 200  \
                        --aug emotion --norm znorm --shift 1

python3 training_data_preprocess.py --input_spec_size 128 --dataset crema-d \
                        --feature_type mel_spec --validate 1 --win_len 200  \
                        --aug emotion --norm znorm --shift 1

python3 training_data_preprocess.py --input_spec_size 128 --dataset msp-improv \
                        --feature_type mel_spec --validate 1 --win_len 200  \
                        --aug emotion --norm znorm --shift 1


python3 adversary_data_preprocess.py --input_spec_size 128 --dataset iemocap \
                        --feature_type mel_spec --validate 1 --win_len 200  \
                        --aug emotion --norm znorm --shift 1 

python3 adversary_data_preprocess.py --input_spec_size 128 --dataset crema-d \
                        --feature_type mel_spec --validate 1 --win_len 200  \
                        --aug emotion --norm znorm --shift 1 

python3 adversary_data_preprocess.py --input_spec_size 128 --dataset msp-improv \
                        --feature_type mel_spec --validate 1 --win_len 200  \
                        --aug emotion --norm znorm --shift 1 