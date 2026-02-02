import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pre_len = configs.pre_len
        self.his_len = configs.his_len
        self.seq_len = self.his_len + self.pre_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu_lLama}"
        print(self.device)

        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir)
        self.hidden_dim_of_gpt2 = 768
        self.mix = configs.mix_embeds
        self.dropout = nn.Dropout(configs.dropout)
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.seq_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.pre_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.seq_len, self.hidden_dim_of_gpt2,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.pre_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) 
    
    def forecast(self, x_enc, x_mark_enc):

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.seq_len, step=self.pre_len)
        times_embeds = self.encoder(fold_out)
        times_embeds = self.dropout(times_embeds)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        outputs = self.gpt2(
            inputs_embeds=times_embeds).last_hidden_state

        outputs = self.dropout(outputs)
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)

        return dec_out
    
    def forward(self, x_enc, x_mark_enc):
        return self.forecast(x_enc, x_mark_enc)
