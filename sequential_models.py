import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel
from transformers import StoppingCriteria, StoppingCriteriaList
from torch import cuda, LongTensor, FloatTensor
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

import ensembles

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class FlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor):
        B, T, _ = x.size()
        
        # Автоматическое создание causal mask
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        
        qkv = self.qkv_proj(x).split(self.embed_dim, dim=2)
        q, k, v = [y.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for y in qkv]

        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,  # Автоматическая маска
                dropout_p=self.dropout if self.training else 0,
                scale=self.scale,
                is_causal=True,  # Оптимизация для Flash Attention
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.masked_fill(causal_mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.training:
                attn = F.dropout(attn, p=self.dropout)
            attn_output = attn @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(attn_output)

class GPTLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ff_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # Attention block
        x = x + self.dropout1(self.attn(self.norm1(x)))
        # FFN block
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x
        
# Тестирование слоя
def test_layer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    layer = GPTLayer(
        dim=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1
    ).to(device)  # Важно: переносим весь слой на устройство
    
    x = torch.randn(2, 10, 512).to(device)
    out = layer(x)
    
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    print("Тест пройден успешно!")
    return x, out

# x, out = test_layer()
# print(x.shape, out.shape)
# x, out

class GPTAdapterLayer(nn.Module):
    def __init__(self, initial_layer, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, conservativity: float = 0.01, t_layers_count = 2,
                net_dropout_rate:float = 0, layer_configs=[2048, 2048], memnet_params={'num_heads':6, 'query_size':64, 'num_key_values':128, 'value_size':256}):
        super().__init__()
        self.initial_layer = initial_layer
        self.conservativity = conservativity
        self.corrector_stack = torch.nn.ModuleList([])
        self.t_layers_count = t_layers_count
        self.transformer_float_mode = 32
        self.lmhead_float_mode = 16
        for i in range(t_layers_count):
            corrector_block = torch.nn.ModuleList([])
            corrector_block += [GPTLayer(
                dim,
                num_heads,
                ff_dim,
                dropout
            )]
            corrector_block += [ensembles.EResNetPro(input_size=dim, 
                   out_size=dim, 
                   net_dropout_rate=net_dropout_rate, 
                   individ_dropout_rate=dropout,
                   layer_configs=layer_configs, 
                   use_sigmoid_end=False, 
                   use_batchnorm=True, 
                   use_activation=True, 
                   activation=nn.LeakyReLU(), 
                   sample_features=1., 
                   composition_size=1, 
                   memnet_params=memnet_params,
                   use_memnets=True)]
            self.corrector_stack += [corrector_block]
    def forward(self, x: Tensor, attention_mask:Tensor, position_ids:Tensor, past_key_value, output_attentions, use_cache, cache_position, position_embeddings):
        out_original = self.initial_layer(x, 
                                          attention_mask=attention_mask, 
                                          position_ids=position_ids,
                                          past_key_value=past_key_value,
                                          output_attentions=output_attentions,
                                          use_cache=use_cache,
                                          cache_position=cache_position,
                                          position_embeddings=position_embeddings)
        out_original_tns = out_original[0]
        if len(out_original) == 2:
            out_original_cache = out_original[1]
        else:
            out_original_cache = None
        del out_original
        if self.transformer_float_mode == 32:
            x = x.to(torch.float32)
            out_original_tns =  out_original_tns.to(torch.float32)
        for i in range(self.t_layers_count):
            corrector_block = self.corrector_stack[i]
            x = corrector_block[0](x) + x
            x = corrector_block[1](x) + x
        x = out_original_tns + x * self.conservativity
        if self.lmhead_float_mode == 16:
            x = x.to(torch.float16)
        else:
            x = x.to(torch.float32)
        return (x, out_original_cache)