import os
import sys

sys.path.append("/home/mw/project")
sys.path.append("/home/mw/project/my_code")

import numpy as np
import random
import torch
import torch.nn.functional as F
from models import iTransformer_iFlashFormer
from index_param import get_param


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def invoke(inputs):
    cwd = os.path.dirname(inputs)
    save_path = "/home/mw/project"

    test_data_root_path = inputs
    data_temp = np.load(
        os.path.join(test_data_root_path, "temp_lookback.npy")
    )  # (N, L, S, 1)
    data_wind = np.load(os.path.join(test_data_root_path, "wind_lookback.npy"))
    N, L, S, _ = data_temp.shape  # 71, 168, 60

    cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))
    # cenn_era5_data[:, :, 3, :, :] = cenn_era5_data[:, :, 3, :, :] / 1e4

    repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1)  # (N, L, 4, 9, S)
    C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
    covariate = repeat_era5.reshape(
        repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]
    )  # (N, L, C1, S)
    for idx in range(0, 165, 3):
        temp_a = (covariate[:, idx, ...] + covariate[:, idx + 3, ...]) / 3
        covariate[:, idx + 1, ...] = temp_a
        covariate[:, idx + 2, ...] = 2 * temp_a
    data_temp = data_temp.transpose(0, 1, 3, 2)  # (N, L, 1, S)
    data_wind = data_wind.transpose(0, 1, 3, 2)  # (N, L, 1, S)
    C = 38
    data = np.concatenate([covariate, data_temp, data_wind], axis=2)  # (N, L, C, S)
    data = data.transpose(0, 3, 1, 2)  # (N, S, L, C)
    data = data.reshape(N * S, L, C)
    data = torch.tensor(data).float()  # (N * S, L, C)
    # data process
    diff = torch.diff(
        data[..., -2:],
        n=1,
        dim=1,
        prepend=torch.zeros((data.size(0), 1, 2), device=data.device, dtype=data.dtype),
    )  # [bs, seq_len, 2]
    rolling_avg = F.avg_pool1d(
        data[..., -2:].permute(0, 2, 1),
        kernel_size=7,
        stride=1,
        padding=3,
        count_include_pad=False,
    ).permute(
        0, 2, 1
    )  # [bs, seq_len, 2]
    wind_scaler = torch.sqrt(
        data[..., :9] ** 2 + data[..., 9:18] ** 2
    )  # [bs, seq_len, 9]
    wind_scaler_mean = wind_scaler.mean(dim=-1, keepdim=True)
    nine_mean = torch.stack(
        [data[..., i : i + 9].mean(dim=-1) for i in range(0, 36, 9)]
    ).permute(
        1, 2, 0
    )  # [bs, seq_len, 4]
    data = torch.cat(
        [
            diff,
            rolling_avg,
            wind_scaler,
            wind_scaler_mean,
            nine_mean,
            data,
        ],
        dim=-1,
    )

    outputs_temps = []
    outputs_winds = []
    N_TTA = 9
    for seed in [3407]:
        for task in ["temp", "wind", "both"]:
            args = get_param("iTransformer_iFlashFormer")
            set_seed(seed)
            args["task"] = task

            class Struct:
                def __init__(self, **entries):
                    self.__dict__.update(entries)

            args = Struct(**args)
            model = iTransformer_iFlashFormer.Model(args).cuda()
            model.eval()

            ckpt_files = os.listdir(f"/home/mw/project/best_model/{task}")

            for ckpt_file in ckpt_files:
                outputs_temp = []
                outputs_wind = []

                ckpt = torch.load(f"/home/mw/project/best_model/{task}/{ckpt_file}")
                new_dict = {}
                for key, value in ckpt.items():
                    if "module.module." in key:
                        new_key = key[14:]
                    elif "module." in key:
                        new_key = key[7:]
                    else:
                        new_key = key
                    new_dict[new_key] = value
                if "n_averaged" in new_dict:
                    new_dict.pop("n_averaged")
                model.load_state_dict(new_dict)
                if task == "both":
                    for iiii in range(0, len(data), 1024):
                        pred = model(data[iiii : iiii + 1024, ...].cuda())
                        pred_temp = pred[0].detach().cpu().numpy()  # (N * S, P, 1)
                        pred_wind = pred[1].detach().cpu().numpy()  # (N * S, P, 1)
                        outputs_temp.append(pred_temp)
                        outputs_wind.append(pred_wind)
                    outputs_temp = np.concatenate(outputs_temp, axis=0)
                    outputs_wind = np.concatenate(outputs_wind, axis=0)
                    P = outputs_temp.shape[1]
                    forecast_temp = outputs_temp.reshape(N, S, P, 1)  # (N, S, P, 1)
                    forecast_temp = forecast_temp.transpose(0, 2, 1, 3)  # (N, P, S, 1)
                    forecast_wind = outputs_wind.reshape(N, S, P, 1)  # (N, S, P, 1)
                    forecast_wind = forecast_wind.transpose(0, 2, 1, 3)  # (N, P, S, 1)
                    outputs_temps.append(forecast_temp)
                    outputs_winds.append(forecast_wind)
                    # TTA (data [bs, seq_len, 56])
                    for _ in range(N_TTA):
                        outputs_temp = []
                        outputs_wind = []
                        for iiii in range(0, len(data), 1024):
                            input_x = data[iiii : iiii + 1024, ...]
                            stds = torch.std(input_x, dim=1, keepdim=True) * 0.1
                            input_x = input_x + torch.normal(0, stds).to(input_x.dtype)
                            pred = model(input_x.cuda())
                            pred_temp = pred[0].detach().cpu().numpy()  # (N * S, P, 1)
                            pred_wind = pred[1].detach().cpu().numpy()  # (N * S, P, 1)
                            outputs_temp.append(pred_temp)
                            outputs_wind.append(pred_wind)
                        outputs_temp = np.concatenate(outputs_temp, axis=0)
                        outputs_wind = np.concatenate(outputs_wind, axis=0)
                        P = outputs_temp.shape[1]
                        forecast_temp = outputs_temp.reshape(N, S, P, 1)  # (N, S, P, 1)
                        forecast_temp = forecast_temp.transpose(
                            0, 2, 1, 3
                        )  # (N, P, S, 1)
                        forecast_wind = outputs_wind.reshape(N, S, P, 1)  # (N, S, P, 1)
                        forecast_wind = forecast_wind.transpose(
                            0, 2, 1, 3
                        )  # (N, P, S, 1)
                        outputs_temps.append(forecast_temp)
                        outputs_winds.append(forecast_wind)
                elif task == "temp":
                    for iiii in range(0, len(data), 1024):
                        pred = model(data[iiii : iiii + 1024, ...].cuda())
                        pred_temp = pred.detach().cpu().numpy()  # (N * S, P, 1)
                        outputs_temp.append(pred_temp)
                    outputs_temp = np.concatenate(outputs_temp, axis=0)
                    P = outputs_temp.shape[1]
                    forecast_temp = outputs_temp.reshape(N, S, P, 1)  # (N, S, P, 1)
                    forecast_temp = forecast_temp.transpose(0, 2, 1, 3)  # (N, P, S, 1)
                    outputs_temps.append(forecast_temp)
                    # TTA (data [bs, seq_len, 56])
                    for _ in range(N_TTA):
                        outputs_temp = []
                        for iiii in range(0, len(data), 1024):
                            input_x = data[iiii : iiii + 1024, ...]
                            stds = torch.std(input_x, dim=1, keepdim=True) * 0.1
                            input_x = input_x + torch.normal(0, stds).to(input_x.dtype)
                            pred = model(input_x.cuda())
                            pred_temp = pred.detach().cpu().numpy()  # (N * S, P, 1)
                            outputs_temp.append(pred_temp)
                        outputs_temp = np.concatenate(outputs_temp, axis=0)
                        P = outputs_temp.shape[1]
                        forecast_temp = outputs_temp.reshape(N, S, P, 1)  # (N, S, P, 1)
                        forecast_temp = forecast_temp.transpose(
                            0, 2, 1, 3
                        )  # (N, P, S, 1)
                        outputs_temps.append(forecast_temp)
                elif task == "wind":
                    for iiii in range(0, len(data), 1024):
                        pred = model(data[iiii : iiii + 1024, ...].cuda())
                        pred_wind = pred.detach().cpu().numpy()  # (N * S, P, 1)
                        outputs_wind.append(pred_wind)
                    outputs_wind = np.concatenate(outputs_wind, axis=0)
                    P = outputs_wind.shape[1]
                    forecast_wind = outputs_wind.reshape(N, S, P, 1)  # (N, S, P, 1)
                    forecast_wind = forecast_wind.transpose(0, 2, 1, 3)  # (N, P, S, 1)
                    outputs_winds.append(forecast_wind)
                    # TTA (data [bs, seq_len, 56])
                    for _ in range(N_TTA):
                        outputs_wind = []
                        for iiii in range(0, len(data), 1024):
                            input_x = data[iiii : iiii + 1024, ...]
                            stds = torch.std(input_x, dim=1, keepdim=True) * 0.1
                            input_x = input_x + torch.normal(0, stds).to(input_x.dtype)
                            pred = model(input_x.cuda())
                            pred_wind = pred.detach().cpu().numpy()  # (N * S, P, 1)
                            outputs_wind.append(pred_wind)
                        outputs_wind = np.concatenate(outputs_wind, axis=0)
                        P = outputs_wind.shape[1]
                        forecast_wind = outputs_wind.reshape(N, S, P, 1)  # (N, S, P, 1)
                        forecast_wind = forecast_wind.transpose(
                            0, 2, 1, 3
                        )  # (N, P, S, 1)
                        outputs_winds.append(forecast_wind)

    results_temp = np.mean(outputs_temps, axis=0)  # (N, P, S, 1)
    results_wind = np.mean(outputs_winds, axis=0)  # (N, P, S, 1)

    np.save(os.path.join(save_path, "temp_predict.npy"), results_temp)
    np.save(os.path.join(save_path, "wind_predict.npy"), results_wind)
