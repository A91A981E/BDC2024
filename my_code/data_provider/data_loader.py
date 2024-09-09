import torch
import os
import numpy as np
from torch.utils.data import Dataset, Subset
import warnings
from tqdm import tqdm
import h5py
from scipy import stats
import gc

warnings.filterwarnings("ignore")


def make_train_val(root_path, train_val_size, size, seed, clean):
    data_temp = np.load(os.path.join(root_path, "temp.npy")).astype(
        np.float32
    )  # (T, S, 1)
    data_temp = data_temp.transpose((0, 2, 1))  # (T 1 S)
    data_wind = np.load(os.path.join(root_path, "wind.npy")).astype(
        np.float32
    )  # (T, S, 1)
    data_wind = data_wind.transpose((0, 2, 1))  # (T 1 S)
    era5 = np.load(os.path.join(root_path, "global_data.npy")).astype(np.float32)
    repeat_era5 = np.repeat(era5, 3, axis=0)  # (T, 4, 9, S)

    repeat_era5 = repeat_era5.reshape(
        repeat_era5.shape[0], -1, repeat_era5.shape[3]
    )  # (T, 36, S)

    data = np.concatenate(
        [repeat_era5, data_temp, data_wind], axis=1, dtype=np.float32
    )  # (T, 38, S)
    # 清洗数据：去掉全0的station
    S = data.shape[-1]
    if clean:
        mask = (np.std(data_temp, axis=0) == 0) | (
            np.std(data_wind, axis=0).squeeze() == 0
        )
        mask = ~mask.squeeze()
        data = data[..., mask]  # (T, 38, S)

        S = data.shape[-1]
    del repeat_era5, data_temp, data_wind
    gc.collect()

    idx = torch.randperm(S)
    train_station = idx[: int(S * train_val_size)]
    # val_station = idx[int(S * train_val_size) :]
    train_set = Dataset_Meteorology_New(data[..., train_station], size=size)
    # val_set = Dataset_Meteorology_New(data[..., val_station], size=size)

    if clean:
        # 去除异常值样本
        assert os.path.exists(f"/home/mw/project/tmp/train_subset_idx_{seed}.hdf5")
        if os.path.exists(f"/home/mw/project/tmp/train_subset_idx_{seed}.hdf5"):
            with h5py.File(f"/home/mw/project/tmp/train_subset_idx_{seed}.hdf5", "r") as f:
                train_subset_idx = f[f"subset_idx"][:]
            train_subset_idx = list(train_subset_idx)
        else:
            train_subset_idx = []
            for train_idx in tqdm(range(len(train_set))):
                x, y = train_set[train_idx]
                if judge(x, y):
                    train_subset_idx.append(train_idx)
            with h5py.File(f"/home/mw/project/tmp/train_subset_idx_{seed}.hdf5", "w") as f:
                f.create_dataset(f"subset_idx", data=train_subset_idx)
        # assert os.path.exists(f"tmp/val_subset_idx_{seed}.hdf5")
        # if os.path.exists(f"tmp/val_subset_idx_{seed}.hdf5"):
        #     with h5py.File(f"tmp/val_subset_idx_{seed}.hdf5", "r") as f:
        #         val_subset_idx = f[f"subset_idx"][:]
        #     val_subset_idx = list(val_subset_idx)
        # else:
        #     val_subset_idx = []
        #     for val_idx in tqdm(range(len(val_set))):
        #         x, y = val_set[val_idx]
        #         if judge(x, y):
        #             val_subset_idx.append(val_idx)
        #     with h5py.File(f"tmp/val_subset_idx_{seed}.hdf5", "w") as f:
        #         f.create_dataset(f"subset_idx", data=val_subset_idx)

        gc.collect()
        return Subset(train_set, train_subset_idx)  # , Subset(val_set, val_subset_idx)
    
    gc.collect()
    train_subset_idx = torch.randperm(len(train_set))[:int(0.95 * len(train_set))]
    return Subset(train_set, train_subset_idx)  # , val_set


def judge(seq_x, seq_y):
    data_temp = np.concatenate([seq_x[:, -2], seq_y[:, -2]], axis=0)
    v = data_temp.var()
    if v < 2.7 or v > 43.2:
        return False
    a = stats.mode(data_temp, keepdims=True)[0][0]
    if (data_temp == a).sum() / 240 > 0.1745:
        return False

    data_wind = np.concatenate([seq_x[:, -1], seq_y[:, -1]], axis=0)
    v = data_wind.var()
    if v < 0.59 or v > 12.2:
        return False
    a = stats.mode(data_wind, keepdims=True)[0][0]
    if (data_wind == a).sum() / 240 > 0.373:
        return False
    return True


class Dataset_Meteorology_New(Dataset):
    def __init__(self, data, size=None):
        self.data = data  # (T, 38, S)
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.stations_num = self.data.shape[-1]
        self.time_num = self.data.shape[0]
        self.tot_len = self.time_num - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):

        sid = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end, :, sid]  # (L1, 36)
        seq_y = self.data[r_begin:r_end, :, sid]  # (L2, 36)

        return seq_x, seq_y[:, -2:]

    def __len__(self):
        return self.tot_len * self.stations_num
