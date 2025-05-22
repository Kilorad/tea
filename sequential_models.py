import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel
from transformers import StoppingCriteria, StoppingCriteriaList
from torch import cuda, LongTensor, FloatTensor
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

import ensembles

from transformers.models.llama.modeling_llama import LlamaRMSNorm as OriginalLlamaRMSNorm
import torch.nn as nn
import torch

class PatchedLlamaRMSNorm(OriginalLlamaRMSNorm):
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # Если размерность скрытых состояний больше весов
        if hidden_states.size(-1) > self.weight.size(0):
            prefix_size = self.weight.size(0)
            
            # Разделяем на две части
            base_part = hidden_states[..., :prefix_size]
            extra_part = hidden_states[..., prefix_size:]
            
            # Обрабатываем только базовую часть
            variance = base_part.pow(2).mean(-1, keepdim=True)
            base_part = base_part * torch.rsqrt(variance + self.variance_epsilon)
            processed = self.weight * base_part.to(input_dtype)
            
            # Собираем обратно
            return torch.cat([processed, extra_part.to(input_dtype)], dim=-1)
            
        # Стандартная обработка
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Монопатчим класс в transformers
import transformers
transformers.models.llama.modeling_llama.LlamaRMSNorm = PatchedLlamaRMSNorm

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
                #attn_mask=causal_mask,  # Автоматическая маска
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
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, conservativity = 0.0001):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ff_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.final_projection = nn.Linear(dim, dim, dropout)
        with torch.no_grad():
            self.final_projection.weight *= conservativity
            self.final_projection.bias *= conservativity

    def forward(self, x: Tensor):
        # Attention block
        x = x + self.dropout1(self.attn(self.norm1(x)))
        # FFN block
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return self.final_projection(x)
        
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
                net_dropout_rate:float = 0, layer_configs=[2048, 2048], memnet_params={'num_heads':8, 'query_size':128, 'num_key_values':200, 'value_size':256}, composition_type='parallel', concat=False):
        #composition_type='parallel' or 'stack'
        super().__init__()
        self.initial_layer = initial_layer
        self.conservativity = conservativity
        self.corrector_stack = torch.nn.ModuleList([])
        self.t_layers_count = t_layers_count
        self.transformer_float_mode = 32
        self.lmhead_float_mode = 16
        self.composition_type = composition_type
        self.concat = concat
        corrector_block = torch.nn.ModuleList([])
        for i in range(t_layers_count):
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
            corrector_block += [GPTLayer(
                dim,
                num_heads,
                ff_dim,
                dropout,
                conservativity
            )]
            self.corrector_stack += [corrector_block]
    def forward(self, x: Tensor, attention_mask:Tensor=None, position_ids:Tensor=None, past_key_value=None, output_attentions=None, use_cache=None, cache_position=None, position_embeddings=None):
        if self.initial_layer is not None:
            out_original = self.initial_layer(x, 
                                              attention_mask=attention_mask, 
                                              position_ids=position_ids,
                                              past_key_value=past_key_value,
                                              output_attentions=output_attentions,
                                              use_cache=use_cache,
                                              cache_position=cache_position,
                                              position_embeddings=position_embeddings)
        else:
            out_original = [x, False]
        out_original_tns = out_original[0]
        if self.composition_type == 'stack':
            x = out_original_tns
        if len(out_original) == 2:
            out_original_cache = out_original[1]
        else:
            out_original_cache = None
        del out_original
        if self.transformer_float_mode == 32:
            out_original_tns =  out_original_tns.to(torch.float32).detach()
            #разрыв градиента. Но пофигу, мы не хотим, чтобы он проходил в предыдущие слои
            x = x.to(torch.float32)
        for i in range(self.t_layers_count):
            corrector_block = self.corrector_stack[i]
            x = x + corrector_block[0](x)
            x = x + corrector_block[1](x)
        if self.concat:
            #есть подозрение, что трансформерные слои в конце модели будут её, эту модель, портить и уродовать. И LM-head не прочтёт это безобразие.
            #поэтому теперь LM-head сможет получить на вход и старый выходной сигнал, и новый - оба.
            x = torch.dstack([x, out_original_tns])
        else:
            x = x + out_original_tns
        if self.lmhead_float_mode == 16:
            x = x.to(torch.float16)
        else:
            if x.dtype != torch.float32:
                #разрыв градиента. Но если у нас x.dtype != torch.float32, то нас это уже не волнует, трансформер работает чисто на инференс
                x = x.to(torch.float32)
        return (x, out_original_cache)


class GPTAdapterLayerWide(nn.Module):
    def __init__(self, initial_layer, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, conservativity: float = 0.01, t_layers_count = 2,
                net_dropout_rate:float = 0, layer_configs=[2048, 2048], memnet_params={'num_heads':8, 'query_size':64, 'num_key_values':200, 'value_size':256}, composition_type='parallel', concat=False, in_adapter_sz=4096):
        '''Здесь особенность в том, что он конкатенирует выходы всех слоёв, в соответствии с идеологией моего resnet'''
        #composition_type='parallel' or 'stack'
        super().__init__()
        self.initial_layer = initial_layer
        self.conservativity = conservativity
        self.corrector_stack = torch.nn.ModuleList([])
        self.t_layers_count = t_layers_count
        self.transformer_float_mode = 32
        self.lmhead_float_mode = 16
        self.composition_type = composition_type
        self.concat = concat
        self.dim = dim

        self.in_adapter_sz = in_adapter_sz#это адаптер, чтобы изменить входную размерность
        self.input_adapter = nn.Linear(self.in_adapter_sz, dim).to(device)
        
        corrector_block = torch.nn.ModuleList([])
        for i in range(t_layers_count):
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
            corrector_block += [GPTLayer(
                dim,
                num_heads,
                ff_dim,
                dropout,
                conservativity
            )]
            self.corrector_stack += [corrector_block]
    def get_output_size(self):
        sz = self.dim * self.t_layers_count + self.in_adapter_sz
        return sz
    def forward(self, x: Tensor, attention_mask:Tensor=None, position_ids:Tensor=None, past_key_value=None, output_attentions=None, use_cache=None, cache_position=None, position_embeddings=None):
        y = []
        if self.initial_layer is not None:
            out_original = self.initial_layer(x, 
                                              attention_mask=attention_mask, 
                                              position_ids=position_ids,
                                              past_key_value=past_key_value,
                                              output_attentions=output_attentions,
                                              use_cache=use_cache,
                                              cache_position=cache_position,
                                              position_embeddings=position_embeddings)
        else:
            out_original = [x, False]
        out_original_tns = out_original[0]
        if self.composition_type == 'stack':
            x = out_original_tns
        if len(out_original) == 2:
            out_original_cache = out_original[1]
        else:
            out_original_cache = None
        del out_original
        if self.transformer_float_mode == 32:
            out_original_tns =  out_original_tns.to(torch.float32).detach()
            #разрыв градиента. Но пофигу, мы не хотим, чтобы он проходил в предыдущие слои
            x = x.to(torch.float32)

        x = self.input_adapter(x)#мы так можем из тонкого трансформера сделать толстый и наоборот
        for i in range(self.t_layers_count):
            corrector_block = self.corrector_stack[i]
            x = x + torch.utils.checkpoint.checkpoint(corrector_block[0], x)
            #y += [x]
            x = x + torch.utils.checkpoint.checkpoint(corrector_block[1], x)
            y += [x]

        #есть подозрение, что трансформерные слои в конце модели будут её, эту модель, портить и уродовать. И LM-head не прочтёт это безобразие.
        #поэтому теперь LM-head сможет получить на вход и старый выходной сигнал, и новый - оба.
        #upd. В конкретно этом адаптере мы на выход хреначим и выходной сигнал исходного трансформера, и всех других слоёв, если их много. Всё через concat
        x = torch.dstack([out_original_tns] + y)
        if self.lmhead_float_mode == 16:
            x = x.to(torch.float16)
        else:
            if x.dtype != torch.float32:
                #разрыв градиента. Но если у нас x.dtype != torch.float32, то нас это уже не волнует, трансформер работает чисто на инференс
                x = x.to(torch.float32)
        return (x, out_original_cache)
def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen
#теперь сборка модели с адаптерами
def assemble_model(model, 
                   path2lmhead, 
                   path2tadapter, 
                   start_train,
                   to_generate,
                   lm_head_adapter_params,
                   transformer_adapter_params,
                   optimizer_params
                  ):
    if start_train:
        print('creating tadapter')
        #инициализировать всё это дело
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        if ('wide' in transformer_adapter_params) and (transformer_adapter_params['wide']):
            tadapter = GPTAdapterLayerWide(initial_layer=None, 
                                     dim=transformer_adapter_params['embed_dim'], 
                                     num_heads=transformer_adapter_params['num_heads_tlayer'], 
                                     ff_dim=transformer_adapter_params['ff_dim'], 
                                     t_layers_count=transformer_adapter_params['t_layers_count'], 
                                     layer_configs=transformer_adapter_params['layer_configs'],
                                     conservativity=transformer_adapter_params['transformer_adapter_weight'],
                                     dropout=transformer_adapter_params['dropout'],
                                     composition_type='stack',
                                     concat=transformer_adapter_params['concat'],
                                     in_adapter_sz=transformer_adapter_params['original_transformer_size']).to(device)
        else:
            tadapter = GPTAdapterLayer(initial_layer=None, 
                                     dim=transformer_adapter_params['embed_dim'], 
                                     num_heads=transformer_adapter_params['num_heads_tlayer'], 
                                     ff_dim=transformer_adapter_params['ff_dim'], 
                                     t_layers_count=transformer_adapter_params['t_layers_count'], 
                                     layer_configs=transformer_adapter_params['layer_configs'],
                                     conservativity=transformer_adapter_params['transformer_adapter_weight'],
                                     dropout=transformer_adapter_params['dropout'],
                                     composition_type='stack',
                                     concat=transformer_adapter_params['concat']).to(device)
    else:
        print('loading tadapter')
        tadapter = torch.load(path2tadapter, weights_only=False)

    
    if ('recreate_lm_head' in lm_head_adapter_params and lm_head_adapter_params['recreate_lm_head']) or start_train:
        print('creating heavy LM head')
        if ('wide' in transformer_adapter_params) and (transformer_adapter_params['wide']):
            #в этом случае 'embedding_size' уже тупо посчитан снаружи. Формула:
            #transformer_adapter_params['embed_dim'] * transformer_adapter_params['t_layers_count'] + transformer_adapter_params['original_transformer_size']
            #Ну типа исходный трансформер + все слои адаптера через конкат
            input_size = lm_head_adapter_params['embedding_size']
            lin_model_size = transformer_adapter_params['original_transformer_size']
        else:
            #размер эмбеддинга lm-head получется равен или размеру эмбеддинга исходного трансформера, или удвоенному, 
            #есkи исходный конкатенируется с адаптерным
            input_size = lm_head_adapter_params['embedding_size'] * (1 + transformer_adapter_params['concat'])
            lin_model_size = lm_head_adapter_params['embedding_size']

        
        head = ensembles.EResNetPro(input_size=input_size, 
                   out_size=lm_head_adapter_params['cardinality'], 
                   net_dropout_rate=lm_head_adapter_params['net_dropout_rate'], 
                   individ_dropout_rate=lm_head_adapter_params['individ_dropout_rate'],
                   layer_configs=lm_head_adapter_params['layer_configs_head'], 
                   use_sigmoid_end=False, 
                   use_batchnorm=True, 
                   use_activation=True, 
                   activation=nn.LeakyReLU(), 
                   sample_features=lm_head_adapter_params['sample_features'], 
                   composition_size=lm_head_adapter_params['composition_size'], 
                   lin_bottleneck_size=None,
                   lin_model_add=nn.Linear(lin_model_size, lm_head_adapter_params['cardinality']).to(device),
                   memnet_params=lm_head_adapter_params['memnet_params'],
                   use_memnets=lm_head_adapter_params['use_memnets'],
                   max_batch_size=lm_head_adapter_params['head_max_batch_size'],
                   aggregation_by_mean=False,
                   exponential_layer_size=False).to(device)
        head.submodels[-1].weight = torch.nn.Parameter(torch.load("lin_model.pth").to(device).to(torch.float32))
        head.submodels[-1].weight.requires_grad = lm_head_adapter_params['learnable_linear_model']
    else:
        print('loading heavy LM head')
        head = torch.load(path2lmhead, weights_only=False)
    
    model.concat = transformer_adapter_params['concat']
    if transformer_adapter_params['concat']:
        if ('wide' in transformer_adapter_params) and (transformer_adapter_params['wide']):
            head.submodels[-1].features = list(range(transformer_adapter_params['original_transformer_size']))
        else:
            head.submodels[-1].features = list(range(transformer_adapter_params['embed_dim']))
    
    #собрать
    model.eval()

    head.to(device)
    head.by_submodels = False

    optimizer = None
    if to_generate:
        #ща сошьём
        head.half()
        tadapter.half()
        head.training = False
        model.lm_head = head
        initial_layer = model.model.layers[-1]
        model.model.layers[-1] = tadapter
        model.model.layers[-1].transformer_float_mode = 16
        model.model.layers[-1].lmhead_float_mode = 16
        model.model.layers[-1].initial_layer = initial_layer
        return
    else:
        head.train()
        head.training = True
        tadapter.train()
        if optimizer_params['opt_type'] == 'adam':
            optimizer = torch.optim.Adam([
                {'params': head.parameters(), 'lr': optimizer_params['lr']},
                {'params': tadapter.parameters(), 'lr': optimizer_params['lr_transformer']}
            ], lr=optimizer_params['lr'])
        else:
            momentum = 0.9
            lr = optimizer_params['lr'] * 0.08
            lr_transformer = optimizer_params['lr_transformer'] * 0.08
            optimizer = torch.optim.SGD([
                {'params': head.parameters(), 'lr': lr},
                {'params': tadapter.parameters(), 'lr': lr_transformer}
            ],  momentum=momentum)
        trainable, frozen = count_trainable_parameters(head)
        print(f"Head. Обучаемые: {trainable:,}\nЗамороженные: {frozen:,}")
        trainable, frozen = count_trainable_parameters(tadapter)
        print(f"Transformer adapter. Обучаемые: {trainable:,}\nЗамороженные: {frozen:,}")
        return head, tadapter, optimizer
def disassemble_model(model):
    #удалить из модели адаптеры
    model.model.layers[-1] = model.model.layers[-1].initial_layer
    model.lm_head = model.lm_head.submodels[-1]