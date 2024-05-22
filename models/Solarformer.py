import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.Pyraformer_EncDec import Encoder
from layers.causal import TCN
from layers.FilterMechanism import FilterMechanism


class Model(nn.Module):

    def __init__(self, configs, window_size=[4, 4], inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.encoder = Encoder(configs, window_size, inner_size)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(
                (len(window_size) + 1) * self.d_model, self.pred_len * configs.enc_in)

        self.mask = FilterMechanism()  

        self.tcn = TCN(input_size=configs.enc_in, num_channels=[16, 32, 64, 128], kernel_size=7,
                       dropout=configs.dropout, out_size=configs.c_out)


    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_mask, dec_mask, mask=None):

        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        enc_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)

        dec_out = torch.cat((enc_out, x_dec[:, :-self.pred_len, :]), dim=1)
        dec_out = self.mask(dec_mask, dec_out)
        dec_out = self.tcn(dec_out)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_mask, dec_mask, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_mask, dec_mask, )
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
