import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, AutoModel, GPT2TokenizerFast

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
        print(self.device)
        # LlamaForCausalLM"则更明确地暗示了模型的用途，即用于因果语言建模。这意呩着这个特定的类被训练用来继承自"LlamaModel”来生成文本，通常是按照单向的、因果的顺序。
        self.gpt2 = GPT2Model.from_pretrained(

            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.GPT2Tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        self.GPT2Tokenizer.pad_token = self.GPT2Tokenizer.eos_token   # 指定填充标记(pad_token)使用结束标记(eos_token)。
        # 使用 max_length=10 和 tokenizer.pad_token = tokenizer.eos_token 进行处理,结果会是: 这是个例子[SEP][SEP][SEP]
        self.vocab_size = self.GPT2Tokenizer.vocab_size    # vocab_size: 32000
        self.hidden_dim_of_llama = 768
        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.GPT2Tokenizer(x, return_tensors="pt")['input_ids'].to(self.device) # x是'This is Time Series from 2016-07-11 16:00:00 to 2016-07-15 15:00:00' 也就是对这句话进行token化，而不是对这两个时间戳之间的所有时间戳
        result = self.gpt2.get_input_embeddings()(output)
        return result   
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        text_outputs = self.gpt2(inputs_embeds=x_mark_enc)[0]    # (128,47,4096)
        # text_outputs = torch.mean(text_outputs, dim = 1)    # llama-7B模型：hidden_size=4096,intermediate_size=11008,num_hidden_layers=32
        text_outputs = text_outputs[:, -1, :]     # llama-7B模型：hidden_size=4096,intermediate_size=11008,num_hidden_layers=32
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)