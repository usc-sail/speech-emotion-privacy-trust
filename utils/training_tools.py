import numpy as np
import torch
import random


emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
gender_dict = {'F': 0, 'M': 1}

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, data_dict, dict_keys, mode='train', input_channel=1):
        """
        Read the textfile and get the paths
        """
        self.data_dict = data_dict
        self.dict_keys = dict_keys
        self.input_channel = input_channel
        self.mode = mode

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.dict_keys[idx]]
        # if self.mode != 'train':
        import pdb
        # pdb.set_trace()

        if self.input_channel == 1:
            specgram = np.expand_dims(data['data'][0], axis=0)
        else:
            specgram = data['data']
        lens = specgram.shape[1]
        
        global_data = data['global_data'][0]
        emo_id = emo_dict[data['label']]
        gen_id = gender_dict[data['gender']]
        sample = {'spec': torch.from_numpy(np.ascontiguousarray(specgram)),
                  'labels_emo': torch.from_numpy(np.ascontiguousarray(emo_id)),
                  'labels_gen': torch.from_numpy(np.ascontiguousarray(gen_id)),
                  'lengths': torch.from_numpy(np.ascontiguousarray(lens)),
                  'global': torch.from_numpy(np.ascontiguousarray(global_data)),}
        return sample

def speech_collate(batch):
    gender = []
    emotion=[]
    specs = []
    lengths = []
    global_data = []
    for sample in batch:
        specs.append(sample['spec'])
        emotion.append((sample['labels_emo']))
        gender.append(sample['labels_gen'])
        lengths.append(sample['lengths'])
        global_data.append(sample['global'])
    return specs, emotion, gender, lengths, global_data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
