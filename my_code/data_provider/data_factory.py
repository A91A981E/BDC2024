from data_provider.data_loader import make_train_val
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data._utils.collate import default_collate

from sklearn.model_selection import KFold


def data_provider(args):

    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size

    # train_set, test_set = make_train_val(
    #     root_path=args.root_path,
    #     train_val_size=args.train_val_size,
    #     size=[args.seq_len, 1, args.pred_len],
    #     seed=args.seed,
    #     clean=args.clean_data,
    # )
    train_set = make_train_val(
        root_path=args.root_path,
        train_val_size=args.train_val_size,
        size=[args.seq_len, 1, args.pred_len],
        seed=args.seed,
        clean=args.clean_data,
    )

    train_data_loaders = []
    val_data_loaders = []
    train_idx_overall = torch.randperm(len(train_set))
    # val_idx_overall = torch.randperm(len(test_set))

    k_fold = KFold(n_splits=args.k_fold)
    for train_idx, val_idx in k_fold.split(train_set):
        train_subset = Subset(train_set, train_idx_overall[val_idx])
        # val_subset = Subset(train_set, train_idx_overall[val_idx])

        train_data_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda batch: collate_fn(batch),
        )
        # val_data_loader = DataLoader(
        #     test_set,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=args.num_workers,
        #     drop_last=False,
        #     collate_fn=lambda batch: collate_fn(batch),
        # )
        train_data_loaders.append(train_data_loader)
        # val_data_loaders.append(val_data_loader)
    return train_data_loaders  # , val_data_loaders


def collate_fn(batch):
    batch_x, batch_y = default_collate(batch)
    # batch_x [bs, seq_len, 38], batch_y [bs, pred_len, 2]
    for idx in range(0, 165, 3):
        temp_a = (batch_x[:, idx, :-2] + batch_x[:, idx + 3, :-2]) / 3
        batch_x[:, idx + 1, :-2] = temp_a
        batch_x[:, idx + 2, :-2] = 2 * temp_a
    diff = torch.diff(
        batch_x[..., -2:],
        n=1,
        dim=1,
        prepend=torch.zeros(
            (batch_x.size(0), 1, 2), device=batch_x.device, dtype=batch_x.dtype
        ),
    )  # [bs, seq_len, 2]
    rolling = F.avg_pool1d(
        batch_x[..., -2:].permute(0, 2, 1), kernel_size=7, stride=1, padding=3
    ).permute(
        0, 2, 1
    )  # [bs, seq_len, 2]
    wind_scaler = torch.stack(
        [torch.sqrt(batch_x[..., i] ** 2 + batch_x[..., i + 9] ** 2) for i in range(9)]
    ).permute(
        1, 2, 0
    )  # [bs, seq_len, 9]
    wind_scaler_mean = wind_scaler.mean(dim=-1, keepdim=True)  # [bs, seq_len, 1]
    nine_mean = torch.stack(
        [batch_x[..., i : i + 9].mean(dim=-1) for i in range(0, 36, 9)]
    ).permute(
        1, 2, 0
    )  # [bs, seq_len, 4]
    batch_x = torch.cat(
        [diff, rolling, wind_scaler, wind_scaler_mean, nine_mean, batch_x], dim=-1
    )  # 2 + 2 + 9 + 1 + 4 + 36 + 2 = 56
    batch_x = torch.nan_to_num(batch_x, nan=0.0)  # [bs, seq_len, 56]
    stds = torch.std(batch_x, dim=1, keepdim=True) * 0.1
    batch_x = batch_x + torch.normal(0.0, stds).to(batch_x.device, batch_x.dtype)
    return batch_x, batch_y
