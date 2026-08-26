import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
from utils.timefeatures import time_features
import datetime
from pathlib import Path

warnings.filterwarnings('ignore')


class Dataset_our(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=False, seasonal_patterns=None, drop_short=False, sp_root=None):
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
        self.sp_root = sp_root
        self.data_path = str(Path(self.root_path) / self.flag / data_path)
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        # Stage-II BOLDCast reads the subject-level NPZ written by the unified
        # preprocessing pipeline and enriched by Stage I.
        with np.load(self.data_path, allow_pickle=True) as temp:
            if 'fMRI' not in temp.files or 'com' not in temp.files:
                raise KeyError(
                    f"{self.data_path} must contain fMRI and com. "
                    f"Available keys: {list(temp.files)}"
                )
            private_key = 'priv' if 'priv' in temp.files else ('pri' if 'pri' in temp.files else None)
            if private_key is None:
                raise KeyError(
                    f"{self.data_path} is missing Stage-I private latent key 'priv' "
                    f"(legacy alias 'pri' is also accepted). Run Stage I first."
                )
            fMRI = torch.as_tensor(temp['fMRI']).float()
            com = torch.as_tensor(temp['com']).float()
            priv = torch.as_tensor(temp[private_key]).float()

        if com.ndim != 2 or priv.ndim != 2:
            raise ValueError(
                f"Expected com/priv to be 2-D [ROI, latent_dim], got "
                f"com={tuple(com.shape)}, priv={tuple(priv.shape)} in {self.data_path}"
            )
        if com.shape[0] != fMRI.shape[0] or priv.shape[0] != fMRI.shape[0]:
            raise ValueError(
                f"ROI mismatch in {self.data_path}: fMRI={tuple(fMRI.shape)}, "
                f"com={tuple(com.shape)}, priv={tuple(priv.shape)}"
            )

        # Concatenated Stage-I representation used only by BOLDCast.
        # Shape: [ROI, common_dim + private_dim].
        self.latent_embed = torch.cat([com, priv], dim=-1)

        self.max = torch.unsqueeze(torch.max(fMRI, dim=1)[0], 1)
        self.min = torch.unsqueeze(torch.min(fMRI, dim=1)[0], 1)
        denom = self.max - self.min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        fMRI = (fMRI - self.min) / denom
        fMRI = torch.where(torch.isnan(fMRI), torch.zeros_like(fMRI), fMRI)

        data_name = Path(self.data_path).stem
        if self.sp_root is None:
            # Backward-compatible fallback: sibling ts/ and sp/ folders.
            ts_root = Path(self.root_path).resolve()
            self.sp_root = str(ts_root.parent / 'sp')
        text_path = Path(self.sp_root) / self.flag / f"{data_name}.npz.pt"
        if not text_path.exists():
            raise FileNotFoundError(f"Timestamp embedding not found: {text_path}")
        self.data_stamp = torch.load(text_path, map_location='cpu')
        self.data_x = fMRI.T
        self.data_y = fMRI.T

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
        # r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.latent_embed



    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        # return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=False, features='M', target='OT', drop_short=True, freq = 's'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.drop_short = drop_short
        self.data_path = str(Path(self.root_path) / self.flag / data_path)
        self.freq = freq
        self.__read_data__()

    def __read_data__(self):
        temp = np.load(self.data_path)
        df_stamp = temp['text']
        temp = torch.tensor(temp['fMRI'])
        data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.max = torch.unsqueeze(torch.mean(temp, dim=1), 1)
        self.min = torch.unsqueeze(torch.std(temp, dim=1), 1)
        temp = torch.div((temp - self.max), (self.min))
        self.data_x = temp.T
        self.data_y = temp.T
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        # temp = dd.io.load(self.root_path + self.data_path)
        temp = np.load(self.root_path + self.data_path)
        self.data_stamp = temp['text']
        self.data_stamp = [str(x) for x in self.data_stamp] # self.data_stamp是list的形式


    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len   # 96
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y/%m/%d %H:%M:%S")    # start格式2016-07-01 00:00:00
        if self.data_set_type in ['traffic', 'electricity', 'ETTh1', 'ETTh2']:
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y/%m/%d %H:%M:%S")  # end格式2016-07-04 23:00:00 datetime.timedelta就是往后数96-1个小时，找到对应的时间戳
        elif self.data_set_type == 'weather':
            end = (start + datetime.timedelta(minutes=10*(self.token_len-1))).strftime("%Y/%m/%d %H:%M:%S")
        elif self.data_set_type in ['ETTm1', 'ETTm2']:
            end = (start + datetime.timedelta(minutes=15*(self.token_len-1))).strftime("%Y/%m/%d %H:%M:%S")
        else:
            end = (start + datetime.timedelta(seconds= self.token_len-1)).strftime("%Y/%m/%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)