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
                self.layers[-1].weight *= 0.0001
                self.layers[-1].bias *= 0.0001
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
            self.layers[-1].weight *= 0.0001
            self.layers[-1].bias *= 0.0001
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
                self.layers[-1].weight *= 0.0001
                self.layers[-1].bias *= 0.0001
            self.layers.append(activation)
            self.layers.append(MemLayer(out_sz, out_sz, num_heads=num_heads, query_size=query_size, value_size=value_size, num_key_values=num_key_values))
            with torch.no_grad():
                self.layers[-1].out_linear.weight *= 0.0001
            self.layers.append(nn.Dropout(p=self.dropout_rate))
            self.layers.append(nn.LayerNorm(out_sz))
        
        if len(layer_configs) > 0:
            in_sz = sum(layer_configs)
        else:
            in_sz = out_sz
        out_sz = out_size
        self.layers.append(nn.Linear(in_sz, out_sz))
        with torch.no_grad():
            self.layers[-1].weight *= 0.0001
            self.layers[-1].bias *= 0.0001
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
                if value < l/6.:
                    value = int(l/6.)
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


class MOE(nn.Module):
    """
    Mixture of Experts (MOE) с поддержкой возврата выходов субмоделей и обработкой больших батчей.
    
    Args:
        input_size (int): Размер входных признаков
        out_size (int): Размер выходного слоя
        dropout_rate (float): Вероятность дропаута
        layer_configs (list): Конфигурация слоёв для экспертов
        router_layer_configs (list): Конфигурация слоёв для роутера
        use_sigmoid_end (bool): Использовать сигмоиду на выходе
        use_batchnorm (bool): Использовать BatchNorm
        use_activation (bool): Использовать активации
        activation (nn.Module): Тип активации
        sample_features (float): Доля используемых признаков
        use_memnets (bool): Использовать MemLayer
        memnet_params (dict): Параметры MemLayer
        exponential_layer_size (bool): Экспоненциальный размер слоёв
        initial_num_experts (int): Начальное количество экспертов
        top_k (int): Число активных экспертов
        inference_top_k (int): Число активных экспертов на инференсе
        lin_bottleneck_size (int): Размер бутылочного горлышка для линейного эксперта
        lin_model_add (nn.Module): Дополнительная линейная модель
        by_submodels (bool): Возвращать выходы всех субмоделей
        max_batch_size (int): Максимальный размер батча для обработки
    """
    def __init__(self, input_size, out_size, dropout_rate, layer_configs=None,
                 router_layer_configs=None, use_sigmoid_end=True, use_batchnorm=True, 
                 use_activation=True, activation=nn.ReLU(), sample_features=0.9, 
                 use_memnets=False, memnet_params=None, exponential_layer_size=True, 
                 initial_num_experts=0, top_k=2, inference_top_k=None,
                 lin_bottleneck_size=None, lin_model_add=None,
                 by_submodels=False, max_batch_size=10000, unlock_last_model=False):
        super().__init__()
        
        # Сохраняем параметры
        self.input_size = input_size
        self.out_size = out_size
        self.dropout_rate = dropout_rate
        self.layer_configs = layer_configs or []
        self.router_layer_configs = router_layer_configs or [32]
        self.use_sigmoid_end = use_sigmoid_end
        self.use_batchnorm = use_batchnorm
        self.use_activation = use_activation
        self.activation = activation
        self.sample_features = sample_features
        self.use_memnets = use_memnets
        self.memnet_params = memnet_params or {}
        self.exponential_layer_size = exponential_layer_size
        self.lin_bottleneck_size = lin_bottleneck_size
        self.lin_model_add = lin_model_add
        self.by_submodels = by_submodels
        self.max_batch_size = max_batch_size
        #это чтобы можно было установить предобученную линейную модель в композицию
        self.unlock_last_model = unlock_last_model
        
        # Параметры MOE
        self.top_k = top_k
        self.inference_top_k = inference_top_k if inference_top_k is not None else top_k
        self.input_size_sampled = min(int(sample_features * input_size) + 1, input_size)
        
        # Инициализация экспертов
        self.submodels = nn.ModuleList()
        self.router = None
        
        # Добавляем начальных экспертов
        for _ in range(initial_num_experts):
            self.add_expert()
        
        # Добавляем линейных экспертов
        self._add_linear_experts()
    
    def _add_linear_experts(self):
        """Добавляет линейных экспертов (lin_bottleneck_size и lin_model_add)"""
        if self.lin_bottleneck_size is not None:
            linear_expert = nn.Sequential(
                nn.Linear(self.input_size, self.lin_bottleneck_size),
                self.activation,
                nn.Linear(self.lin_bottleneck_size, self.out_size)
            )
            linear_expert.features = list(range(self.input_size))
            self.add_expert(linear_expert)
        
        if self.lin_model_add is not None:
            if not hasattr(self.lin_model_add, 'features'):
                self.lin_model_add.features = list(range(self.input_size))
            self.add_expert(self.lin_model_add)
    
    def to(self, device, *args, **kwargs):
        """Перемещает модель на устройство"""
        super().to(device, *args, **kwargs)
        if self.router is not None:
            self.router = self.router.to(device)
        return self
    
    def add_expert(self, expert=None):
        """Добавляет нового эксперта в ансамбль"""
        if expert is None:
            # Создаем нового эксперта со случайным набором признаков
            features_set = random.sample(range(self.input_size), self.input_size_sampled)
            
            # Создаем конфигурацию слоёв для эксперта
            layer_configs_current = []
            for l in self.layer_configs:
                if self.exponential_layer_size:
                    value = int(np.random.exponential(scale=l))
                else:
                    value = l
                if value < 3:
                    value = 3
                layer_configs_current.append(value)
            
            # Создаем экземпляр эксперта
            if self.use_memnets:
                expert = ResMemNet(
                    self.input_size_sampled, 
                    self.out_size, 
                    self.dropout_rate,
                    layer_configs_current,
                    False,
                    self.use_batchnorm,
                    self.use_activation,
                    self.activation,
                    **self.memnet_params
                )
            else:
                expert = ResNet(
                    self.input_size_sampled, 
                    self.out_size, 
                    self.dropout_rate,
                    layer_configs_current,
                    False,
                    self.use_batchnorm,
                    self.use_activation,
                    self.activation
                )
            
            # Сохраняем набор признаков
            expert.features = features_set
        else:
            # Используем переданного эксперта
            if not hasattr(expert, 'features'):
                expert.features = random.sample(range(self.input_size), self.input_size_sampled)
        
        # Добавляем эксперта
        self.submodels.append(expert)
        
        # Перестраиваем роуте
        self._rebuild_router()
    
    def _rebuild_router(self):
        """Перестраивает роутер для текущего числа экспертов"""
        num_experts = len(self.submodels)
        
        if num_experts == 0:
            self.router = None
            return
        
        # Создаем роутер как ResNet
        self.router = ResNet(
            input_size=self.input_size,
            out_size=num_experts,
            dropout_rate=self.dropout_rate,
            layer_configs=self.router_layer_configs,
            use_sigmoid_end=False,
            use_bathcnorm=self.use_batchnorm,
            use_activation=self.use_activation,
            activation=self.activation
        )
    
    def forward(self, X):
        if len(self.submodels) == 0:
            return torch.zeros((X.size(0), self.out_size)), [] if self.by_submodels else None
        
        # Определяем количество активных экспертов
        active_k = self.top_k if self.training else self.inference_top_k
        active_k = min(active_k, len(self.submodels))
        
        # Разбиваем батч на подбатчи
        outputs = []
        all_submodels_outputs = []
        
        for i in range(0, X.size(0), self.max_batch_size):
            #batch_slice = slice(i, min(i + self.max_batch_size, X.size(0)))
            X_batch = X[i: i + self.max_batch_size]
            
            # Вычисляем для подбатча
            batch_output, batch_submodels_outputs = self._forward_batch(X_batch, active_k)
            
            outputs.append(batch_output)
            if self.by_submodels:
                all_submodels_outputs.append(batch_submodels_outputs)
        
        # Объединяем результаты
        output = torch.cat(outputs, dim=0)
        
        if self.by_submodels:
            # Объединяем выходы субмоделей
            submodels_outputs = []
            for i in range(len(self.submodels)):
                expert_outputs = [sub[i] for sub in all_submodels_outputs]
                submodels_outputs.append(torch.cat(expert_outputs, dim=0))
            return output, submodels_outputs
        else:
            return output
    
    def _forward_batch(self, X, active_k):
        """Обрабатывает один батч (не более max_batch_size)"""
        batch_size = X.size(0)
        shape = X.shape
        if len(X.shape) == 3:
            X = X.view(shape[0] * shape[1], shape[2])
        num_experts = len(self.submodels)
        
        # Вычисляем веса экспертов
        router_logits = self.router(X)
        weights = F.softmax(router_logits, dim=-1)
        
        
        # Выбираем топ-K экспертов
        if self.unlock_last_model:
            topk_weights, topk_indices = torch.topk(weights[:, :-1], k=active_k, dim=-1, sorted=False)
        else:
            topk_weights, topk_indices = torch.topk(weights, k=active_k, dim=-1, sorted=False)
        
        # Нормализуем веса
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-5)
        # if self.unlock_last_model:
        #     ones = torch.ones((topk_weights.shape[0], 1), 
        #                 device=topk_weights.device, 
        #                 dtype=topk_weights.dtype)
        #     topk_weights = torch.hstack([topk_weights[:, :-1], ones])
        #     ones_ids = torch.ones((topk_indices.shape[0], 1), 
        #                 device=topk_indices.device, 
        #                 dtype=topk_indices.dtype)
        #     topk_indices = torch.hstack([topk_indices, ones_ids * (weights.shape[-1] - 1)])
        
        # Вычисляем выходы всех экспертов, если нужно
        if self.by_submodels:
            all_expert_outputs = torch.zeros((batch_size, num_experts, self.out_size), device=X.device)
            for expert_idx, expert in enumerate(self.submodels):
                expert_input = X[:, expert.features]
                all_expert_outputs[:, expert_idx] = expert(expert_input)
        else:
            all_expert_outputs = None
        
        # Собираем общий выход
        output = torch.zeros((batch_size, self.out_size), device=X.device)
        
        # Группируем вызовы экспертов
        for expert_idx in range(num_experts):
            mask = (topk_indices == expert_idx).any(dim=1)
            
            if not mask.any():
                continue
                
            # Выбираем примеры для этого эксперта
            expert_input = X[mask]
            expert = self.submodels[expert_idx]
            
            # Выбираем нужные признаки
            expert_input = expert_input[:, expert.features]
            
            # Вычисляем выход эксперта
            if self.by_submodels:
                expert_output = all_expert_outputs[mask, expert_idx]
            else:
                expert_output = expert(expert_input)
            
            # Получаем веса
            expert_weights = topk_weights[mask, (topk_indices[mask] == expert_idx).nonzero()[:, 1]]
            
            # Взвешенно суммируем
            output[mask] += expert_weights.unsqueeze(1) * expert_output

        if self.unlock_last_model:
            if self.by_submodels:
                expert_output = all_expert_outputs[:, -1]
            else:
                expert = self.submodels[-1]
                expert_input = X[:, expert.features]
                expert_output = expert(expert_input)
            output += expert_output
        
        # Применяем сигмоиду при необходимости
        if self.use_sigmoid_end:
            output = torch.sigmoid(output)
        
        # Возвращаем выходы субмоделей, если нужно
        if len(shape) == 3:
            output = output.view(shape[0], shape[1], output.shape[-1])
        if self.by_submodels:
            submodels_outputs = []
            for i in range(num_experts):
                submodels_outputs.append(all_expert_outputs[:, i])
            return output, submodels_outputs
        else:
            return output, None
    
    def set_inference_top_k(self, k):
        """Устанавливает число активных экспертов для инференса"""
        self.inference_top_k = k