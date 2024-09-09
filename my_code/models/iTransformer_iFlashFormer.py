import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
    FlashAttention,
)
from layers.Embed import DataEmbedding_inverted
from models.decoder import Decoder_Varb
from models.MMoE import Gate


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len, configs.d_model, configs.dropout
        )
        # Encoder
        self.encoder = nn.ModuleList(
            [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(
                                    False,
                                    attention_dropout=configs.dropout,
                                    output_attention=False,
                                ),
                                configs.d_model,
                                configs.n_heads,
                            ),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                        )
                        for _ in range(configs.e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(configs.d_model),
                ),
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FlashAttention(
                                    False,
                                    configs.factor,
                                    attention_dropout=configs.dropout,
                                    output_attention=False,
                                ),
                                configs.d_model,
                                configs.n_heads,
                            ),
                            configs.d_model,
                            configs.d_ff,
                            dropout=configs.dropout,
                            activation=configs.activation,
                        )
                        for _ in range(configs.e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(configs.d_model),
                ),
            ]
        )
        assert len(self.encoder) == configs.n_experts
        self.n_task = 2 if configs.task == "both" else 1
        self.task = configs.task
        self.n_experts = configs.n_experts
        self.gate = nn.ModuleList(
            [Gate(configs.d_model, configs.n_experts) for _ in range(self.n_task)]
        )
        if configs.task == "both":
            self.projection_temp = Decoder_Varb(
                in_dim=configs.d_model,
                pred_len=configs.pred_len,
                dropout=configs.dropout,
                n_var=configs.enc_in,
            )
            self.projection_wind = Decoder_Varb(
                in_dim=configs.d_model,
                pred_len=configs.pred_len,
                dropout=configs.dropout,
                n_var=configs.enc_in,
            )
        else:
            self.projection = Decoder_Varb(
                in_dim=configs.d_model,
                pred_len=configs.pred_len,
                dropout=configs.dropout,
                n_var=configs.enc_in,
            )

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding
        enc_out_embed = self.enc_embedding(x_enc, None)  # [B, N, D]
        enc_out = torch.stack(
            [
                self.encoder[i](enc_out_embed, attn_mask=None)[0]
                for i in range(self.n_experts)
            ]
        )  # [E, B, N, D]

        # MMoE
        if self.n_task == 2:
            temp_embed = enc_out_embed[:, -2, :]
            wind_embed = enc_out_embed[:, -1, :]
            g = torch.stack(
                [self.gate[0](temp_embed), self.gate[1](wind_embed)]
            )  # [T, B, E]
            enc_out = torch.einsum("ebvd,tbe->tbvd", enc_out, g)  # [T, B, N, D]

            # Decode
            dec_t = self.projection_temp(enc_out[0, ...], dim=-2)  # [B, P]
            dec_w = self.projection_wind(enc_out[1, ...], dim=-1)  # [B, P]

            # De-Normalization from Non-stationary Transformer
            dec_t = dec_t * (stdev[:, 0, -2].unsqueeze(1).repeat(1, self.pred_len))
            dec_t = dec_t + (means[:, 0, -2].unsqueeze(1).repeat(1, self.pred_len))
            dec_w = dec_w * (stdev[:, 0, -1].unsqueeze(1).repeat(1, self.pred_len))
            dec_w = dec_w + (means[:, 0, -1].unsqueeze(1).repeat(1, self.pred_len))
            return dec_t, dec_w
        else:
            dim = -2 if self.task == "temp" else -1
            embed = enc_out_embed[:, dim, :]
            g = torch.stack([self.gate[0](embed)])
            enc_out = torch.einsum("ebvd,tbe->tbvd", enc_out, g)
            dec_out = self.projection(enc_out[0, ...], dim=dim)
            dec_out = dec_out * (stdev[:, 0, dim].unsqueeze(1).repeat(1, self.pred_len))
            dec_out = dec_out + (means[:, 0, dim].unsqueeze(1).repeat(1, self.pred_len))
            return dec_out

    def forward(self, x_seq):
        dec_out_varb = self.forecast(x_seq)
        return dec_out_varb  # [B, P]
