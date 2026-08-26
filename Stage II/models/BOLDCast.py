import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pre_len = configs.pred_len
        self.his_len = configs.label_len
        self.seq_len = self.his_len + self.pre_len
        self.device = f"cuda:{configs.gpu}" if configs.use_gpu else "cpu"
        print(self.device)

        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir)
        self.hidden_dim_of_gpt2 = 768
        self.mix = configs.mix_embeds
        self.dropout = nn.Dropout(configs.dropout)

        # mix_embeds keeps the original time-embedding fusion behavior and,
        # when enabled, additionally injects the Stage-I common/private latent.
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
            self.latent_scale = nn.Parameter(torch.ones([]))
            # The input dimension is inferred from cat([com, priv], dim=-1), so
            # Stage I can change c_dim/p_dim without adding another CLI argument.
            self.latent_projection = nn.LazyLinear(self.hidden_dim_of_gpt2)

        for _, param in self.gpt2.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.seq_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.pre_len)
        else:
            print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(
                self.seq_len,
                self.hidden_dim_of_gpt2,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation,
            )
            self.decoder = MLP(
                self.hidden_dim_of_gpt2,
                self.pre_len,
                configs.mlp_hidden_dim,
                configs.mlp_hidden_layers,
                configs.dropout,
                configs.mlp_activation,
            )

    def _prepare_time_embed(self, x_mark_enc, bs, n_vars, token_num):
        """Align subject-level timestamp embeddings with ROI-level GPT2 tokens."""
        if x_mark_enc.ndim != 3 or x_mark_enc.shape[0] != bs:
            raise ValueError(
                f"Expected x_mark_enc [B, tokens, 768], got {tuple(x_mark_enc.shape)}"
            )
        if x_mark_enc.shape[-1] != self.hidden_dim_of_gpt2:
            raise ValueError(
                f"Timestamp embedding dim must be {self.hidden_dim_of_gpt2}, "
                f"got {x_mark_enc.shape[-1]}"
            )
        if x_mark_enc.shape[1] != token_num:
            x_mark_enc = F.adaptive_avg_pool1d(
                x_mark_enc.transpose(1, 2), token_num
            ).transpose(1, 2)
        x_mark_enc = F.normalize(x_mark_enc, dim=-1)
        return x_mark_enc.repeat_interleave(n_vars, dim=0)

    def _prepare_latent_embed(self, latent_embed, bs, n_vars, token_num):
        """Project Stage-I latent [B, ROI, D] to GPT2 token space."""
        if latent_embed is None:
            raise ValueError(
                "mix_embeds=True requires latent_embed from Stage I (com + priv)."
            )
        if latent_embed.ndim != 3:
            raise ValueError(
                f"Expected latent_embed [B, ROI, D], got {tuple(latent_embed.shape)}"
            )
        if latent_embed.shape[0] != bs or latent_embed.shape[1] != n_vars:
            raise ValueError(
                f"latent_embed shape {tuple(latent_embed.shape)} does not match "
                f"BOLDCast input batch/ROI dimensions ({bs}, {n_vars})."
            )

        latent_tokens = self.latent_projection(latent_embed.float())
        latent_tokens = F.normalize(latent_tokens, dim=-1)
        latent_tokens = latent_tokens.reshape(bs * n_vars, 1, self.hidden_dim_of_gpt2)
        if token_num != 1:
            latent_tokens = latent_tokens.expand(-1, token_num, -1)
        return latent_tokens

    def forecast(self, x_enc, x_mark_enc, latent_embed=None):
        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.seq_len, step=self.pre_len)
        times_embeds = self.encoder(fold_out)
        times_embeds = self.dropout(times_embeds)

        if self.mix:
            # Preserve the original timestamp-embedding fusion.
            times_embeds = F.normalize(times_embeds, dim=-1)
            time_tokens = self._prepare_time_embed(
                x_mark_enc, bs=bs, n_vars=n_vars, token_num=times_embeds.shape[1]
            )
            times_embeds = times_embeds + self.add_scale * time_tokens

            # Add Stage-I common/private representation using the same switch.
            latent_tokens = self._prepare_latent_embed(
                latent_embed,
                bs=bs,
                n_vars=n_vars,
                token_num=times_embeds.shape[1],
            )
            times_embeds = times_embeds + self.latent_scale * latent_tokens

        outputs = self.gpt2(inputs_embeds=times_embeds).last_hidden_state
        outputs = self.dropout(outputs)
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, latent_embed=None):
        return self.forecast(x_enc, x_mark_enc, latent_embed)
