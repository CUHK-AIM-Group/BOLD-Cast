import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings
import deepdish as dd

warnings.filterwarnings('ignore')


class Dataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='',
                 scale=False, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = self.root_path + self.flag + '/' + data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        temp = np.load(self.data_path)
        fMRI = torch.tensor(temp['fMRI'])

        self.max = torch.unsqueeze(torch.max(fMRI, dim=1)[0], 1)
        self.min =  torch.unsqueeze(torch.min(fMRI, dim=1)[0], 1)
        fMRI = (fMRI - self.min) / (self.max - self.min)

        fMRI = torch.where(torch.isnan(fMRI), torch.full_like(fMRI, 0), fMRI)
        # ukb/hcp
        # data_name = self.data_path[-11:].split('.')[0]
        # abide
        # data_name = self.data_path[-9:].split('.')[0]
        # hcp-a
        data_name = self.data_path[-9:].split('.')[0]
        text_path = self.root_path[:-3] + 'sp/' + self.flag + '/' + data_name + '.npz.pt'
        self.data_stamp = torch.load(text_path)
        self.data_x = fMRI.T
        self.data_y = fMRI.T
        # self.label  = label

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark



    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
