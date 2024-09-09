import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import os


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo")

    # basic config
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )

    # data loader
    parser.add_argument("--clean_data", action="store_true", default=False)
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./best_model/",
        help="location of model checkpoints",
    )
    parser.add_argument("--factor", default=1.0, type=float)
    parser.add_argument(
        "--train_val_size",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--n_experts",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--task", choices=["temp", "wind", "both"], type=str, default="both"
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )

    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")

    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_ckpt", type=str, default="")
    parser.add_argument("--resume_epoch", default=0, type=int)

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1", help="device ids of multile gpus"
    )

    args = parser.parse_args()

    fix_seed = args.seed
    set_seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() else False

    # print("CUDA is available: ", torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Long_Term_Forecast

    print(args)
    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(args)  # set experiments
        exp.train()
