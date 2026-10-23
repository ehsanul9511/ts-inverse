import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding
from .layers.masking import TriangularCausalMask, ProbMask
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    name = "Transformer"

    def __init__(self):
        super(Model, self).__init__()
        self.extra_info = {}
        self.seq_len = 96
        self.pred_len = 96
        self.output_attention = False

        # Multivariate forecasting: preserve all seven channels.
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7

        d_model = 256
        d_ff = 256
        e_layers = 2
        d_layers = 1
        n_heads = 8
        factor = 1
        dropout = 0.1
        embed = "fixed"
        freq = "h"
        activation = "gelu"

        self.enc_embedding = DataEmbedding(
            self.enc_in, d_model, embed, freq, dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.dec_embedding = DataEmbedding(
            self.dec_in, d_model, embed, freq, dropout
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    # Causal self-attention over decoder time steps.
                    AttentionLayer(
                        FullAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    # Cross-attention to encoder representations.
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, self.c_out, bias=True),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        handles = []

        def check_output(name):
            def hook(module, args, output):
                values = output if isinstance(output, (tuple, list)) else (output,)
                for value in values:
                    if (
                        isinstance(value, torch.Tensor)
                        and not torch.isfinite(value).all().item()
                    ):
                        raise RuntimeError(
                            f"First non-finite module output: {name} "
                            f"({type(module).__name__}), "
                            f"shape={tuple(value.shape)}"
                        )
            return hook

        # Hooks inspect outputs without changing them or detaching the graph.
        for name, module in self.named_modules():
            if name:
                handles.append(
                    module.register_forward_hook(check_output(name))
                )

        try:
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=None)

            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(
                dec_out, enc_out, x_mask=None, cross_mask=None
            )
            return dec_out
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]