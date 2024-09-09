from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from data_provider.utils import data_prefetcher
from utils.tools import adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from models.tricks import FGM, PGD
from utils.sam import SAM

import gc

from tqdm import tqdm

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self, **kwargs):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if len(self.args.pretrain_ckpt) > 0:
            ckpt = torch.load(
                os.path.join(
                    self.args.pretrain_ckpt,
                    f"checkpoint_fold_{kwargs['fold_idx']}_3.pth",
                ),
                map_location="cpu",
            )
            new_dict = {}
            for key, value in ckpt.items():
                if "module.module." in key:
                    new_key = key[14:]
                elif "module." in key:
                    new_key = key[7:]
                else:
                    new_key = key
                new_dict[new_key] = value
            model.load_state_dict(new_dict, strict=False)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        # train_data_loaders, val_data_loaders = data_provider(self.args)
        train_data_loaders = data_provider(self.args)
        return train_data_loaders  # , val_data_loaders

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        def loss(a, b):
            return 0.7 * nn.MSELoss()(a, b) + 0.3 * nn.L1Loss()(a, b)

        # criterion = nn.HuberLoss()
        return loss

    def val(self, ema_model, val_prefetcher, criterion):
        ema_model.eval()
        self.model.eval()
        val_loss, val_loss_temp, val_loss_wind = [], [], []
        batch_x, batch_y = val_prefetcher.next()
        with torch.no_grad():
            with tqdm(total=len(val_prefetcher)) as _tqdm:
                while batch_x is not None:
                    with torch.cuda.amp.autocast():
                        output = ema_model(batch_x)
                    if self.args.task == "both":
                        loss_temp = criterion(output[0], batch_y[..., -2])
                        loss_wind = criterion(output[1], batch_y[..., -1])
                        loss = loss_temp + loss_wind
                        val_loss_temp.append(loss_temp.item())
                        val_loss_wind.append(loss_wind.item())
                    elif self.args.task == "temp":
                        loss_temp = criterion(output, batch_y[..., -2])
                        loss = loss_temp
                        val_loss_temp.append(loss_temp.item())
                        val_loss_wind.append(0.0)
                    elif self.args.task == "wind":
                        loss_wind = criterion(output, batch_y[..., -1])
                        loss = loss_wind
                        val_loss_temp.append(0.0)
                        val_loss_wind.append(loss_wind.item())
                    val_loss.append(loss.item())
                    batch_x, batch_y = val_prefetcher.next()
                    _tqdm.update(1)
        ema_model.train()
        self.model.train()
        val_loss = np.average(val_loss)
        val_loss_temp = np.average(val_loss_temp)
        val_loss_wind = np.average(val_loss_wind)
        return val_loss, val_loss_temp, val_loss_wind

    def train(self):
        # train_data_loaders, val_data_loaders = self._get_data()
        train_data_loaders = self._get_data()

        for idx_fold in range(len(train_data_loaders)):
            # print(f"Fold {idx_fold}")
            train_data_loader = train_data_loaders[idx_fold]
            # val_data_loader = val_data_loaders[idx_fold]
            # print("train:", len(train_data_loader), "val:", len(val_data_loader))
            self.model = self._build_model(fold_idx=idx_fold).to(self.device)
            if self.args.clean_data:
                prefix = "_clean_data"
            else:
                prefix = ""
            prefix += f"_{self.args.seed}"
            path = os.path.join(self.args.checkpoints, self.args.task)
            if not os.path.exists(path):
                os.makedirs(path)

            time_now = time.time()

            train_steps = len(train_data_loader)

            model_optim = self._select_optimizer()
            criterion = self._select_criterion()

            pgd = PGD(self.model, k=3)

            scaler = torch.cuda.amp.GradScaler()
            # ema_model = torch.optim.swa_utils.AveragedModel(
            #     self.model,
            #     multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
            #     device=self.device,
            # )
            iter = 0
            for epoch in range(self.args.train_epochs):
                # print(f"Training epoch {epoch + 1}.")
                train_prefetcher = data_prefetcher(train_data_loader, self.device)
                # val_prefetcher = data_prefetcher(val_data_loader, self.device)
                iter_count = 0
                train_loss, train_loss_temp, train_loss_wind = [], [], []
                train_loss_t, train_loss_t_temp, train_loss_t_wind = [], [], []

                # ema_model.train()
                self.model.train()
                epoch_time = time.time()
                batch_x, batch_y = train_prefetcher.next()
                i = 0
                # with tqdm(total=len(train_prefetcher)) as _tqdm:
                if True:
                    while batch_x is not None:
                        iter_count += 1
                        i += 1
                        iter += 1
                        # _tqdm.update(1)
                        self.model.zero_grad()

                        with torch.cuda.amp.autocast():
                            output = self.model(batch_x)

                        if self.args.task == "both":
                            loss_temp = criterion(output[0], batch_y[..., -2])
                            loss_wind = criterion(output[1], batch_y[..., -1])
                            loss = loss_temp + loss_wind
                            train_loss_temp.append(loss_temp.item())
                            train_loss_wind.append(loss_wind.item())
                            train_loss_t_temp.append(loss_temp.item())
                            train_loss_t_wind.append(loss_wind.item())
                        elif self.args.task == "temp":
                            loss = criterion(output, batch_y[..., -2])
                            train_loss_temp.append(loss.item())
                            train_loss_wind.append(0.0)
                            train_loss_t_temp.append(loss.item())
                            train_loss_t_wind.append(0.0)
                        elif self.args.task == "wind":
                            loss = criterion(output, batch_y[..., -1])
                            train_loss_temp.append(0.0)
                            train_loss_wind.append(loss.item())
                            train_loss_t_temp.append(0.0)
                            train_loss_t_wind.append(loss.item())
                        train_loss.append(loss.item())
                        train_loss_t.append(loss.item())

                        scaler.scale(loss).backward()

                        pgd.backup_grad()
                        for t in range(pgd.k):
                            pgd.attack(is_first_attack=(t == 0))
                            if t != pgd.k - 1:
                                self.model.zero_grad()
                            else:
                                pgd.restore_grad()
                            with torch.cuda.amp.autocast():
                                output = self.model(batch_x)
                            if self.args.task == "both":
                                loss_adv_temp = criterion(output[0], batch_y[..., -2])
                                loss_adv_wind = criterion(output[1], batch_y[..., -1])
                                loss_adv = loss_adv_temp + loss_adv_wind
                            elif self.args.task == "temp":
                                loss_adv_temp = criterion(output, batch_y[..., -2])
                                loss_adv = loss_adv_temp
                            elif self.args.task == "wind":
                                loss_adv_wind = criterion(output, batch_y[..., -1])
                                loss_adv = loss_adv_wind
                            scaler.scale(loss_adv).backward()
                        pgd.restore()

                        scaler.step(model_optim)
                        scaler.update()

                        # ema_model.update_parameters(self.model)

                        if i % self.args.print_freq == 0:
                            speed = (time.time() - time_now) / iter_count
                            left_time = (
                                speed
                                * ((self.args.train_epochs - epoch) * train_steps - i)
                                / 60
                            )
                            print(
                                "\nFold: {8}, Epoch: {2}, iters: {0}/{1} | loss: {3:.7f} temp {4:.7f} wind {5:.7f} speed: {6:.4f}s/iter; left time: {7:.4f} min".format(
                                    i + 0,
                                    train_steps,
                                    epoch + 1,
                                    np.average(train_loss_t),
                                    np.average(train_loss_t_temp),
                                    np.average(train_loss_t_wind),
                                    speed,
                                    left_time,
                                    idx_fold,
                                ),
                            )
                            iter_count = 0
                            time_now = time.time()
                            train_loss_t = []
                            train_loss_t_temp = []
                            train_loss_t_wind = []

                        batch_x, batch_y = train_prefetcher.next()
                        gc.collect()

                # # eval epoch
                # torch.optim.swa_utils.update_bn(train_data_loader, ema_model)
                # val_loss, val_loss_temp, val_loss_wind = self.val(
                #     self.model, val_prefetcher, criterion
                # )
                val_loss, val_loss_temp, val_loss_wind = 0, 0, 0

                print(
                    "Epoch: {} cost time: {}".format(
                        epoch + 1, time.time() - epoch_time
                    ),
                    end="",
                )
                train_loss = np.average(train_loss)
                train_loss_temp = np.average(train_loss_temp)
                train_loss_wind = np.average(train_loss_wind)

                print(
                    "\tFold: {8}, Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}, Temp {3:.7f}, "
                    "Wind {4:.7f}, Val Loss: {5:.7f} Temp {6:.7f} Wind {7:.7f}".format(
                        epoch + 1,
                        train_steps,
                        train_loss,
                        train_loss_temp,
                        train_loss_wind,
                        val_loss,
                        val_loss_temp,
                        val_loss_wind,
                        idx_fold,
                    )
                )
                torch.save(
                    self.model.state_dict(),
                    path + "/" + f"checkpoint_fold_{idx_fold}_{epoch + 1}{prefix}.pth",
                )
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            # torch.optim.swa_utils.update_bn(train_data_loader, ema_model)
            # torch.save(
            #     self.model.state_dict(),
            #     path + "/" + f"checkpoint_final_{idx_fold}{prefix}.pth",
            # )
        return self.model
