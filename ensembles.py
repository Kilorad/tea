import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class MemLayer(nn.Module):
    #Слой-база данных
    def __init__(self, input_size, output_size, num_heads, query_size, num_key_values, value_size):
        super(MemLayer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.query_size = query_size
        self.num_key_values = num_key_values
        self.value_size = value_size

        # Обучаемые ключи и значения
        self.keys = nn.Parameter(torch.randn(num_key_values, query_size))
        self.values = nn.Parameter(torch.randn(num_key_values, value_size))

        # Обучаемые параметры для запросов
        self.query_linear = nn.Linear(input_size, num_heads * query_size, bias=False)
        self.out_linear = nn.Linear(num_heads * value_size, output_size, bias=False)

    def forward(self, x):
        batch_size = x.size(0)

        # Генерация запросов
        queries = self.query_linear(x).view(batch_size, self.num_heads, self.query_size)

        # Вычисление внимания
        keys = self.keys.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.num_key_values, self.query_size)  # (batch_size, num_key_values, query_size)
        values = self.values.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.num_key_values, self.value_size)  # (batch_size, num_key_values, value_size)
        # Перемножение с использованием операций
        scores = torch.einsum('aib,ajb->aijb', queries, keys) / (self.query_size ** 0.5)
        # Применяем softmax для весов
        scores = torch.sum(scores, axis=-1)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, num_key_values)

        # Взвешивание значений
        output_values = torch.einsum('aib,ajc->aijc',attn_weights, values)  # (batch_size, num_heads, value_size)
        output_values = torch.mean(output_values, axis=2)
        # Объединение голов
        output_values = output_values.view(batch_size, -1)  # (batch_size, num_heads * value_size)
        output_values = self.out_linear(output_values)
        
        return output_values + x

class ResNet(nn.Module):
    def __init__(self, input_size, out_size, dropout_rate, layer_configs=None, use_sigmoid_end=True, use_bathcnorm=True, use_activation=True, activation=nn.ReLU()):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        out_sz = input_size
        for hidden_sz in layer_configs:
            in_sz = out_sz
            out_sz = hidden_sz
            self.layers.append(nn.Linear(in_sz, out_sz))
            with torch.no_grad():
                self.layers[-1].weight *= 0.01
                self.layers[-1].bias *= 0.01
            self.layers.append(activation)
            self.layers.append(nn.Dropout(p=self.dropout_rate))
            self.layers.append(nn.LayerNorm(out_sz))
        
        if len(layer_configs) > 0:
            in_sz = sum(layer_configs)
        else:
            in_sz = out_sz
        out_sz = out_size
        self.layers.append(nn.Linear(in_sz, out_sz))
        with torch.no_grad():
            self.layers[-1].weight *= 0.01
            self.layers[-1].bias *= 0.01
        self.layers.append(nn.Dropout(p=self.dropout_rate))
        self.layers.append(nn.LayerNorm(out_sz))
        self.layers.append(nn.Sigmoid())
        
        self.out_size = out_size
        self.use_sigmoid_end = use_sigmoid_end
        self.use_batchnorm = use_bathcnorm
        self.use_activation = use_activation
        
        self.activation = activation
        
    def forward(self, X):
        #X = torch.tensor(X, dtype=torch.float16)
        concat_result = []
        i = 0
        for l in self.layers[:-4]:
            if not self.use_batchnorm and ('LayerNorm' in str(l)):
                concat_result.append(X)
                continue
            if not self.use_activation and (str(self.activation) in str(l)):
                continue    
            if ('LayerNorm' in str(l)) and (len(X.shape) == 3):
                shp = X.shape
                X = X.view([shp[0], shp[1] * shp[2]])
                if l._parameters['weight'].shape[0] != X.shape[-1]:
                    self.layers[i] = nn.LayerNorm(X.shape[-1], device=X.device)
                    l = self.layers[i]
                X = l(X)
                X = X.view([shp[0], shp[1], shp[2]])
            else:
                X = l(X)
            if 'LayerNorm' in str(l):
                concat_result.append(X)
            i += 1
        if len(X.shape) == 2:
            X = torch.hstack(concat_result)
        else:
            X = torch.dstack(concat_result)
        
        for l in self.layers[-4:]:
            if not self.use_activation and (str(self.activation) in str(l)):
                continue
            if not self.use_batchnorm and ('LayerNorm' in str(l)):
                continue
            if ('Sigmoid' in str(l)) and (not self.use_sigmoid_end):
                break
            if ('LayerNorm' in str(l)) and (len(X.shape) == 3):
                shp = X.shape
                X = X.view([shp[0], shp[1] * shp[2]])
                if l._parameters['weight'].shape[0] != X.shape[-1]:
                    self.layers[i] = nn.LayerNorm(X.shape[-1], device=X.device)
                    l = self.layers[i]
                X = l(X)
                X = X.view([shp[0], shp[1], shp[2]])
            else:
                X = l(X)
            i += 1
        return X

class ResMemNet(nn.Module):
    def __init__(self, input_size, out_size, dropout_rate, layer_configs=None, use_sigmoid_end=True, use_bathcnorm=True, use_activation=True, activation=nn.ReLU(), num_heads=4, query_size=64, num_key_values=128, value_size=256):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        out_sz = input_size
        for hidden_sz in layer_configs:
            in_sz = out_sz
            out_sz = hidden_sz
            self.layers.append(nn.Linear(in_sz, out_sz))
            with torch.no_grad():
                self.layers[-1].weight *= 0.00001
                self.layers[-1].bias *= 0.00001
            self.layers.append(activation)
            self.layers.append(MemLayer(out_sz, out_sz, num_heads=num_heads, query_size=query_size, value_size=value_size, num_key_values=num_key_values))
            with torch.no_grad():
                self.layers[-1].out_linear.weight *= 0.001
            self.layers.append(nn.Dropout(p=self.dropout_rate))
            self.layers.append(nn.LayerNorm(out_sz))
        
        if len(layer_configs) > 0:
            in_sz = sum(layer_configs)
        else:
            in_sz = out_sz
        out_sz = out_size
        self.layers.append(nn.Linear(in_sz, out_sz))
        with torch.no_grad():
            self.layers[-1].weight *= 0.00001
            self.layers[-1].bias *= 0.00001
        self.layers.append(nn.Dropout(p=self.dropout_rate))
        self.layers.append(nn.LayerNorm(out_sz))
        self.layers.append(nn.Sigmoid())
        
        self.out_size = out_size
        self.use_sigmoid_end = use_sigmoid_end
        self.use_batchnorm = use_bathcnorm
        self.use_activation = use_activation
        
        self.activation = activation
        
    def forward(self, X):
        #X = torch.tensor(X, dtype=torch.float16)
        concat_result = []
        i = 0
        for l in self.layers[:-4]:
            if not self.use_batchnorm and ('LayerNorm' in str(l)):
                concat_result.append(X)
                continue
            if not self.use_activation and (str(self.activation) in str(l)):
                continue    
            if ('LayerNorm' in str(l)) and (len(X.shape) == 3):
                shp = X.shape
                X = X.view([shp[0], shp[1] * shp[2]])
                if l._parameters['weight'].shape[0] != X.shape[-1]:
                    self.layers[i] = nn.LayerNorm(X.shape[-1], device=X.device)
                    l = self.layers[i]
                X = l(X)
                X = X.view([shp[0], shp[1], shp[2]])
            else:
                X = l(X)
            if 'LayerNorm' in str(l):
                concat_result.append(X)
            i += 1
        if len(X.shape) == 2:
            X = torch.hstack(concat_result)
        else:
            X = torch.dstack(concat_result)
        
        for l in self.layers[-4:]:
            if not self.use_activation and (str(self.activation) in str(l)):
                continue
            if not self.use_batchnorm and ('LayerNorm' in str(l)):
                continue
            if ('Sigmoid' in str(l)) and (not self.use_sigmoid_end):
                break
            if ('LayerNorm' in str(l)) and (len(X.shape) == 3):
                continue
                #не нужна
                # shp = X.shape
                # X = X.view([shp[0], shp[1] * shp[2]])
                # if l._parameters['weight'].shape[0] != X.shape[-1]:
                #     self.layers[i] = nn.LayerNorm(X.shape[-1], device=X.device)
                #     l = self.layers[i]
                # X = l(X)
                # X = X.view([shp[0], shp[1], shp[2]])
            else:
                X = l(X)
            i += 1
        return X

def tuple_vstack(Y):
    """
    Аналог torch.vstack для списка тензоров или списка кортежей вида (tensor, [tensor, tensor, ...]).
    
    Args:
        Y: list тензоров или list кортежей вида (tensor, [tensor, tensor, ...])
    
    Returns:
        Если на входе list тензоров - возвращает torch.vstack(Y)
        Если на входе list кортежей - возвращает кортеж (vstacked_main, [vstacked_1, vstacked_2, ...])
    """
    if not Y:
        return Y
    
    # Проверяем первый элемент, чтобы определить формат данных
    first_element = Y[0]
    
    if isinstance(first_element, tuple):
        # Случай кортежей: (tensor, [tensor, tensor, ...])
        main_tensors = []
        list_of_lists = []
        
        # Инициализируем list_of_lists в зависимости от количества тензоров во втором элементе кортежа
        if len(first_element[1]) > 0:
            list_of_lists = [[] for _ in range(len(first_element[1]))]
        
        for item in Y:
            main_tensors.append(item[0])
            for i, sub_tensor in enumerate(item[1]):
                list_of_lists[i].append(sub_tensor)
        
        # Собираем результаты
        stacked_main = torch.vstack(main_tensors)
        stacked_sublist = [torch.vstack(lst) for lst in list_of_lists]
        
        return (stacked_main, stacked_sublist)
    else:
        # Простой случай: list тензоров
        return torch.vstack(Y)
    
class EResNetPro(nn.Module):
    '''
    Вероятностная композиция резнетов. Примечательна тем, что
    1) резнеты разные (их гиперы выбираются из распределения, где большие слои идут с меньшей вероятностью)
    2) иная форма дропаута. Выбрасываются некоторые из резнетов целиком
    '''
    def __init__(self, input_size, out_size, net_dropout_rate, individ_dropout_rate, layer_configs=None, use_sigmoid_end=True, use_batchnorm=True, use_activation=True, activation=nn.ReLU(), sample_features=0.9, composition_size=200, feature_name: str = "features_vec", lin_bottleneck_size=None, lin_model_add=None, use_memnets=False, memnet_params={}, max_batch_size=1024 * 10, aggregation_by_mean=True, exponential_layer_size=True):
        '''теперь мы задаём матожидание размера слоя, а не его фактический размер
        memnet_params={'num_heads', 'query_size', 'num_key_values', 'value_size'}'''
        super().__init__()
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self._feature_count = input_size
        self.use_sigmoid_end = use_sigmoid_end
        self.net_dropout = net_dropout_rate
        self.individ_dropout = individ_dropout_rate
        self.lin_bottleneck_size = lin_bottleneck_size
        self.lin_model_add = lin_model_add
        self.use_memnets = use_memnets
        self.memnet_params = memnet_params
        #aggregation_by_mean обычно надо ставить в true - это логика равноценного бустинга или RF
        #но можно и в false - тогда легче получить ситуацию, что в параллель есть основная модель и вспомогательная
        self.aggregation_by_mean = aggregation_by_mean
        
        self.input_size_sampled = min(int(sample_features * input_size) + 1, input_size)
        
        self.submodels = nn.ModuleList()
        self.by_submodels = False
        self.max_batch_size = max_batch_size
        
        for i in range(composition_size):
            layer_configs_current = []
            for l in layer_configs:
                if exponential_layer_size:
                    value = int(np.random.exponential(scale=l))
                else:
                    value = l
                if value < 3:
                    value = 3
                layer_configs_current.append(value)
            print('use_memnets', self.use_memnets)
            if self.use_memnets:
                self.submodels.append(ResMemNet(self.input_size_sampled, out_size, self.individ_dropout, layer_configs_current, False, use_batchnorm, use_activation, activation, num_heads=self.memnet_params['num_heads'], query_size=self.memnet_params['query_size'], num_key_values=self.memnet_params['num_key_values'], value_size=self.memnet_params['value_size']))
            else:
                self.submodels.append(ResNet(self.input_size_sampled, out_size, self.individ_dropout, layer_configs_current, False, use_batchnorm, use_activation, activation))
                
            features_set = list(range(input_size))
            features_set = random.sample(features_set, self.input_size_sampled)
            self.submodels[-1].features = features_set

        
        if not (self.lin_bottleneck_size is None):
            self.lin_submodel = nn.Sequential(
              nn.Linear(input_size, self.lin_bottleneck_size),
              nn.Linear(self.lin_bottleneck_size, out_size)
            )
            self.submodels.append(self.lin_submodel)
            composition_size += 1
        if not (self.lin_model_add is None):
            self.lin_submodel = self.lin_model_add
            self.submodels.append(self.lin_submodel)
            composition_size += 1
            
        
        self.composition_size = composition_size
        self.output_dimension = out_size
        self.feature_name = feature_name

    def forward(self, X):
        Y = []
        for batch_start in range(0, X.shape[0], self.max_batch_size):
            Y += [self.forward_batch(X[batch_start : batch_start + self.max_batch_size])]
        
        Y = tuple_vstack(Y)
        return Y
    def forward_batch(self, X):
        composition_size_effective = self.composition_size
        if self.training:
            if self.net_dropout <=0:
                idx_drop = torch.zeros(self.composition_size, device=X.device)
            for trial in range(25):
                idx_drop = torch.rand(self.composition_size, device=X.device) #< self.net_dropout
                if torch.all(idx_drop):
                    idx_drop[:] = 0
                idx_drop = idx_drop.to(torch.uint8)
                composition_size_effective = torch.sum( 1 - idx_drop)#то есть делим потом на число актуальных, а не всех, субмоделей
                #мы хотим, чтобы при дропауте была гарантия, что выкинется не менее self.net_dropout субмоделей
                if (composition_size_effective <= self.composition_size * (1 - self.net_dropout) + 0.2) or (self.net_dropout<=0):
                    break
            
            if not (self.lin_bottleneck_size is None):
                idx_drop[-1] = 0 #линейная субмодель жива всегда
            
        #X = X.to(torch.float32)
        Y = None

        Y_lst = []
        for i in range(self.composition_size):
            if self.training:
                if idx_drop[i]:
                    continue
            if 'features' in self.submodels[i].__dict__.keys():
                features_set = self.submodels[i].features
            else:
                features_set = torch.arange(X.shape[1])
                
            X_shape = X.shape
            if len(X_shape) == 3:
                X = X.view(X_shape[0] * X_shape[1], X_shape[2])
            if Y is None:
                Y = self.submodels[i](X[:, features_set]) / composition_size_effective
                if self.by_submodels:
                    Y_lst += [Y.clone()]
            else:
                if self.aggregation_by_mean:
                    Y_add = self.submodels[i](X[:, features_set]) / composition_size_effective
                else:
                    Y_add = self.submodels[i](X[:, features_set])
                Y += Y_add
                if self.by_submodels:
                    Y_lst += [Y_add.clone()]
            if len(X_shape) == 3:
                Y = Y.view(X_shape[0], X_shape[1], Y.shape[-1])
        if self.use_sigmoid_end:
            Y = nn.Sigmoid()(Y)
        if self.by_submodels:
            return Y, Y_lst
        else:
            return Y


class EResNetM(nn.Module):
    '''
    Вероятностная композиция резнетов. Примечательна тем, что
    1) резнеты разные (их гиперы выбираются из распределения, где большие слои идут с меньшей вероятностью)
    2) иная форма дропаута. Выбрасываются некоторые из резнетов целиком

    + менеджер. Смысл менеджера в том, чтобы на инференсе запускать не все субмодели
    '''
    def __init__(self, input_size, out_size, dropout_rate, layer_configs=None, use_sigmoid_end=True, use_batchnorm=True, use_activation=True, activation=nn.ReLU(), sample_features=0.9, composition_size=200, feature_name: str = "features_vec", lin_bottlenenck_size=None):
        '''теперь мы задаём матожидание размера слоя, а не его фактический размер'''
        super().__init__()
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)
        self._feature_count = input_size
        self.use_sigmoid_end = use_sigmoid_end
        self.dropout = dropout_rate
        self.lin_bottlenenck_size = lin_bottlenenck_size
        
        
        self.input_size_sampled = min(int(sample_features * input_size) + 1, input_size)

        self.submodels = nn.ModuleList()
        
        for i in range(composition_size):
            layer_configs_current = []
            for l in layer_configs:
                value = int(np.random.exponential(scale=l))
                if value < 3:
                    value = 3
                layer_configs_current.append(value)
            self.submodels.append(ResNet(self.input_size_sampled, out_size, 0, layer_configs_current, False, use_batchnorm, use_activation, activation))
            features_set = list(range(input_size))
            features_set = random.sample(features_set, self.input_size_sampled)
            self.submodels[-1].features = features_set

        
        if not (self.lin_bottlenenck_size is None):
            self.lin_submodel = nn.Sequential(
              nn.Linear(input_size, self.lin_bottlenenck_size),
              nn.Linear(self.lin_bottlenenck_size, out_size)
            )
            self.submodels.append(self.lin_submodel)
            composition_size += 1

        
        self.manager = ResNet(input_size, composition_size, 0, [input_size//4, input_size//4, input_size//4], True, use_batchnorm, use_activation, activation)
        self.composition_size = composition_size
        self.output_dimension = out_size
        self.feature_name = feature_name
        self.submodels_inference_count = composition_size
    def forward(self, X):
        weights_submodels = self.manager(X)
        composition_size_effective = self.composition_size
        if self.training:
            while 1:
                idx_drop = torch.rand(self.composition_size, device=X.device) < self.dropout
                if torch.all(idx_drop):
                    idx_drop[:] = 0
                idx_drop = idx_drop.to(torch.uint8)
                composition_size_effective = torch.sum( 1 - idx_drop)
                #мы хотим, чтобы при дропауте была гарантия, что выкинется не менее self.dropout субмоделей
                if composition_size_effective <= self.composition_size * (1 - self.dropout):
                    break
            
            if not (self.lin_bottlenenck_size is None):
                idx_drop[-1] = 0 #линейная субмодель жива всегда
            #idx_drop_mx = torch.stack([idx_drop.to(torch.bool)] * weights_submodels.shape[0])
            idx_drop_nums = torch.ravel(torch.nonzero(idx_drop))
            #weights_submodels[:, idx_drop_nums] = 0
        else:
            idx_drop = torch.zeros(self.composition_size, device=X.device)
            #idx_drop_mx = torch.stack([idx_drop.to(torch.bool)] * weights_submodels.shape[0])
            #weights_submodels = torch.nn.functional.softmax(weights_submodels, dim=-1)
            #на инференсе можно сказать, сколько субмоделей инференсить
            if self.submodels_inference_count < self.composition_size:
                weights_submodels_sum = torch.sum(weights_submodels, axis=0)
                _, w_indices = torch.sort(weights_submodels_sum)
                idx_drop[w_indices[self.submodels_inference_count:]] = 1
                idx_drop_nums = torch.ravel(torch.nonzero(idx_drop))
                #weights_submodels[:, idx_drop_nums] = 0
                #дропнуть self.submodels_inference_count субмоделей
            #weights_submodels
            #self.submodels_inference_count
        idx_ndrop_nums = torch.ravel(torch.nonzero(1 - idx_drop))
        #weights_submodels[:, idx_ndrop_nums] = torch.nn.functional.softmax(weights_submodels[:, idx_ndrop_nums], dim=-1)
        #X = X.to(torch.float32)
        Y = None
        
        for i in range(self.composition_size):
            if idx_drop[i]:
                continue
            if 'features' in self.submodels[i].__dict__.keys():
                features_set = self.submodels[i].features
            else:
                features_set = torch.arange(X.shape[1])
                
            X_shape = X.shape
            if len(X_shape) == 3:
                X = X.view(X_shape[0] * X_shape[1], X_shape[2])
            if Y is None:
                Yd = (self.submodels[i](X[:, features_set]) / self.composition_size) 
                Y = Yd * torch.hstack([weights_submodels[:, i:i + 1]] * Yd.shape[1])
            else:
                Yd = (self.submodels[i](X[:, features_set]) / self.composition_size) 
                Y += Yd * torch.hstack([weights_submodels[:, i:i + 1]] * Yd.shape[1])
            if len(X_shape) == 3:
                Y = Y.view(X_shape[0], X_shape[1], Y.shape[-1])
        if self.use_sigmoid_end:
            Y = nn.Sigmoid()(Y)
        return Y
