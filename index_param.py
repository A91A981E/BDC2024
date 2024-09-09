def get_param(model_name):

    if model_name == "iTransformer_iFlashFormer":
        args = {
            "model": "iTransformer_iFlashFormer",
            "checkpoints": "./checkpoints/",
            "seq_len": 168,
            "pred_len": 72,
            "enc_in": 56,
            "d_model": 128,
            "n_heads": 8,
            "e_layers": 1,
            "d_ff": 64,
            "dropout": 0.1,
            "activation": "gelu",
            "n_experts": 2,
            "k_fold": 5,
            "factor": 1.0,
            "task": "both",
        }

    return args
