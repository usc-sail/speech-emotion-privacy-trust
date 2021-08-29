# This is the script for feature extraction

# audio feature part
# three datasets are iemocap, crema-d, and msp-improv dataset
# features include mel spectrogram, mfcc (40), and gemap opensmile features
python3 audio_feature_extraction.py --dataset iemocap --feature_len 128 --feature_type mel_spec
python3 audio_feature_extraction.py --dataset crema-d --feature_len 128 --feature_type mel_spec
python3 audio_feature_extraction.py --dataset msp-improv --feature_len 128 --feature_type mel_spec


