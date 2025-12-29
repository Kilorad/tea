import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import math


class BatchedLinear(nn.Module):
    def __init__(self, in_features, out_features, feature_batch_size, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_batch_size = feature_batch_size
        
        # Main layer parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Weight initialization as in standard Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Save original shape (for multi-dimensional tensor support)
        input_shape = x.shape
        
        # Convert to 2D tensor [N, in_features]
        x_flat = x.view(-1, input_shape[-1])
        batch_size = x_flat.size(0)
        
        # Initialize output tensor
        output_flat = torch.zeros(batch_size, self.out_features, 
                                 device=x.device, dtype=x.dtype)
        
        # Calculate number of feature batches
        num_batches = (self.in_features + self.feature_batch_size - 1) // self.feature_batch_size
        
        # Process each feature batch
        for i in range(num_batches):
            start_idx = i * self.feature_batch_size
            end_idx = min((i + 1) * self.feature_batch_size, self.in_features)
            
            # Select current feature batch from input
            x_batch = x_flat[:, start_idx:end_idx]
            
            # Select corresponding weights
            weight_batch = self.weight[:, start_idx:end_idx]
            
            # Calculate partial result
            output_flat += torch.mm(x_batch, weight_batch.t())
        
        # Add bias (if exists)
        if self.bias is not None:
            output_flat += self.bias
        
        # Restore original shape
        return output_flat.view(*input_shape[:-1], self.out_features)

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'feature_batch_size={self.feature_batch_size}, bias={self.bias is not None}')

class DynamicLinear(nn.Module):
    '''Dynamic linear layer. Sometimes it's necessary to output only some signals, not all, to avoid memory issues.'''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Full layer parameters
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        
        # Active classes register
        self.register_buffer('active_classes', None, persistent=False)
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**0.5 if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def set_active_classes(self, classes):
        """Set active classes externally"""
        if classes is not None:
            classes = torch.as_tensor(classes, dtype=torch.long)
        self.active_classes = classes

    def forward(self, x):
        if self.active_classes is None or not self.training:
            # Full mode: standard linear layer
            return nn.functional.linear(x, self.weight, self.bias)
        
        # Dynamic mode
        active_weight = self.weight[self.active_classes]
        active_bias = self.bias[self.active_classes]
        
        # Calculate only active logits
        active_logits = x @ active_weight.t() + active_bias
        
        # Create zero tensor of full size
        full_logits = torch.zeros(
            x.size(0), 
            self.output_dim,
            dtype=x.dtype,
            device=x.device
        )
        
        # Fill active positions
        full_logits[:, self.active_classes] = active_logits
        
        return full_logits

class MemLayer(nn.Module):
    # Database layer
    def __init__(self, input_size, output_size, num_heads, query_size, num_key_values, value_size):
        super(MemLayer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.query_size = query_size
        self.num_key_values = num_key_values
        self.value_size = value_size

        # Trainable keys and values
        self.keys = nn.Parameter(torch.randn(num_key_values, query_size))
        self.values = nn.Parameter(torch.randn(num_key_values, value_size))

        # Trainable parameters for queries
        self.query_linear = nn.Linear(input_size, num_heads * query_size, bias=False)
        self.out_linear = nn.Linear(num_heads * value_size, output_size, bias=False)

    def forward(self, x):
        batch_size = x.size(0)

        # Generate queries
        queries = self.query_linear(x).view(batch_size, self.num_heads, self.query_size)

        # Calculate attention
        keys = self.keys.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.num_key_values, self.query_size)  # (batch_size, num_key_values, query_size)
        values = self.values.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.num_key_values, self.value_size)  # (batch_size, num_key_values, value_size)
        # Multiplication using operations
        scores = torch.einsum('aib,ajb->aijb', queries, keys) / (self.query_size ** 0.5)
        # Apply softmax for weights
        scores = torch.sum(scores, axis=-1)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, num_key_values)

        # Weighting values
        output_values = torch.einsum('aib,ajc->aijc',attn_weights, values)  # (batch_size, num_heads, value_size)
        output_values = torch.mean(output_values, axis=2)
        # Combine heads
        output_values = output_values.view(batch_size, -1)  # (batch_size, num_heads * value_size)
        output_values = self.out_linear(output_values)
        
        return output_values + x
        
class ResNet(nn.Module):
    def __init__(self, input_size, out_size, dropout_rate, layer_configs=None, use_sigmoid_end=True, use_bathcnorm=True, use_activation=True, activation=nn.ReLU(), bottleneck_sz=None):
        super().__init__()
        self.stop_end_activations = False
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        self.bottleneck_sz = bottleneck_sz
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
        if bottleneck_sz is None:
            self.layers.append(nn.Linear(in_sz, out_sz))
        else:
            self.out_decoder_bn = nn.LayerNorm(in_sz)
            self.out_decoder = nn.Linear(in_sz, bottleneck_sz)
            self.layers.append(nn.Linear(bottleneck_sz, out_sz))
            with torch.no_grad():
                self.out_decoder.weight *= 0.01
                self.out_decoder.bias *= 0.01
        with torch.no_grad():
            self.layers[-1].weight *= 0.1
            self.layers[-1].bias *= 0.1
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
        if not hasattr(self, 'bottleneck_sz'):
            self.bottleneck_sz = None
        if not hasattr(self, 'stop_end_activations'):
            self.stop_end_activations = False
        
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

        if self.bottleneck_sz is not None:
            X = self.out_decoder_bn(X)
            X = self.out_decoder(X)
        
        for l in self.layers[-4:]:
            if not self.use_activation and (str(self.activation) in str(l)):
                continue
            if not self.use_batchnorm and ('LayerNorm' in str(l)):
                continue
            if self.stop_end_activations:
                #Don't need it here
                if ('LayerNorm' in str(l)):
                    continue
                #And this either
                if (str(self.activation) in str(l)):
                    continue
                #And this either
                if ('Dropout' in str(l)):
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
                #not needed
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
    Analogue of torch.vstack for list of tensors or list of tuples of form (tensor, [tensor, tensor, ...]).
    
    Args:
        Y: list of tensors or list of tuples of form (tensor, [tensor, tensor, ...])
    
    Returns:
        If input is list of tensors - returns torch.vstack(Y)
        If input is list of tuples - returns tuple (vstacked_main, [vstacked_1, vstacked_2, ...])
    """
    if not Y:
        return Y
    
    # Check first element to determine data format
    first_element = Y[0]
    
    if isinstance(first_element, tuple):
        # Tuple case: (tensor, [tensor, tensor, ...])
        main_tensors = []
        list_of_lists = []
        
        # Initialize list_of_lists based on number of tensors in second element of tuple
        if len(first_element[1]) > 0:
            list_of_lists = [[] for _ in range(len(first_element[1]))]
        
        for item in Y:
            main_tensors.append(item[0])
            for i, sub_tensor in enumerate(item[1]):
                list_of_lists[i].append(sub_tensor)
        
        # Gather results
        stacked_main = torch.vstack(main_tensors)
        stacked_sublist = [torch.vstack(lst) for lst in list_of_lists]
        
        return (stacked_main, stacked_sublist)
    else:
        # Simple case: list of tensors
        return torch.vstack(Y)
    
class EResNetPro(nn.Module):
    '''
    Probabilistic composition of ResNets. Notable for:
    1) ResNets are different (their hyperparameters are chosen from distribution where larger layers have lower probability)
    2) Different dropout form. Some ResNets are dropped entirely.
    '''
    def __init__(self, input_size, out_size, net_dropout_rate, individ_dropout_rate, layer_configs=None, use_sigmoid_end=True, use_batchnorm=True, use_activation=True, activation=nn.ReLU(), sample_features=0.9, composition_size=200, feature_name: str = "features_vec", lin_bottleneck_size=None, lin_model_add=None, use_memnets=False, memnet_params={}, max_batch_size=1024 * 10, aggregation_by_mean=True, exponential_layer_size=True):
        '''now we set expected layer size, not its actual size
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
        #aggregation_by_mean should usually be set to true - this is logic of equivalent boosting or RF
        #but can be set to false - then it's easier to have main model and auxiliary model in parallel
        self.aggregation_by_mean = aggregation_by_mean
        
        self.input_size_sampled = min(int(sample_features * input_size) + 1, input_size)
        
        self.submodels = nn.ModuleList()
        self.by_submodels = False
        self.max_batch_size = max_batch_size
        if exponential_layer_size:
            random_num = np.random.exponential(scale=1)
        
        for i in range(composition_size):
            layer_configs_current = []
            for l in layer_configs:
                if exponential_layer_size:
                    value = int(random_num * l)
                else:
                    value = l
                if value < l/4.:
                    value = int(l/4.)
                layer_configs_current.append(value)
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
        X_shape = X.shape
        if len(X.shape) == 3:
            X = X.view([X_shape[0] * X_shape[1], X_shape[2]])
        for batch_start in range(0, X.shape[0], self.max_batch_size):
            Y += [self.forward_batch(X[batch_start : batch_start + self.max_batch_size])]
            
        Y = tuple_vstack(Y)
        if len(X_shape) == 3:
            Y = Y.view([X_shape[0], X_shape[1], Y.shape[-1]])
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
                composition_size_effective = torch.sum( 1 - idx_drop)#meaning we divide by actual number, not all, submodels
                #we want guarantee that at least self.net_dropout submodels will be dropped
                if (composition_size_effective <= self.composition_size * (1 - self.net_dropout) + 0.2) or (self.net_dropout<=0):
                    break
            
            if not (self.lin_bottleneck_size is None):
                idx_drop[-1] = 0 #linear submodel is always alive
            
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
                if len(Y.shape) == 3:
                    Y = Y[:, 0, :]
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
    Mixture of Experts (MOE) with support for returning submodel outputs and large batch processing.
    
    Args:
        input_size (int): Input feature size
        out_size (int): Output layer size
        dropout_rate (float): Dropout probability
        layer_configs (list): Layer configuration for experts
        router_layer_configs (list): Layer configuration for router
        use_sigmoid_end (bool): Use sigmoid at output
        use_batchnorm (bool): Use BatchNorm
        use_activation (bool): Use activations
        activation (nn.Module): Activation type
        sample_features (float): Proportion of features to use
        use_memnets (bool): Use MemLayer
        memnet_params (dict): MemLayer parameters
        exponential_layer_size (bool): Exponential layer size
        initial_num_experts (int): Initial number of experts
        top_k (int): Number of active experts
        inference_top_k (int): Number of active experts during inference
        lin_bottleneck_size (int): Bottleneck size for linear expert
        lin_model_add (nn.Module): Additional linear model
        by_submodels (bool): Return outputs of all submodels
        max_batch_size (int): Maximum batch size for processing
    """
    def __init__(self, input_size, out_size, dropout_rate, layer_configs=None,
                 router_layer_configs=None, use_sigmoid_end=True, use_batchnorm=True, 
                 use_activation=True, activation=nn.ReLU(), sample_features=0.9, 
                 use_memnets=False, memnet_params=None, exponential_layer_size=True, 
                 initial_num_experts=0, top_k=2, inference_top_k=None,
                 lin_bottleneck_size=None, lin_model_add=None,
                 by_submodels=False, max_batch_size=10000, unlock_last_model=False, lin_projection_bottleneck=None, resnet_bottleneck_sz=None):
        super().__init__()
        
        # Save parameters
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
        #this is to be able to set pre-trained linear model in composition
        self.unlock_last_model = unlock_last_model
        #this is to go to output not directly, but through bottleneck
        self.lin_projection_bottleneck = lin_projection_bottleneck
        self.lin_projection_bottleneck_block = 256#for some reason large bottlenecks lag terribly, superlinearly. So let's split into blocks.
        self.resnet_bottleneck_sz = resnet_bottleneck_sz
        
        # MOE parameters
        self.top_k = top_k
        self.inference_top_k = inference_top_k if inference_top_k is not None else top_k
        self.input_size_sampled = min(int(sample_features * input_size) + 1, input_size)
        
        # Expert initialization
        self.submodels = nn.ModuleList()
        self.router = None
        
        # Add initial experts
        for _ in range(initial_num_experts):
            self.add_expert()
        
        # Add linear experts
        self._add_linear_experts()
        self.forbidden_tokens_list = []
        self.forbidden_tokens_counter = 0

        if self.lin_projection_bottleneck is not None:              
            self.linear_projector = BatchedLinear(
                in_features=self.lin_projection_bottleneck, 
                out_features=self.out_size,
                feature_batch_size=self.lin_projection_bottleneck_block
            )
    
    def _add_linear_experts(self):
        """Adds linear experts (lin_bottleneck_size and lin_model_add)"""
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
        """Moves model to device"""
        super().to(device, *args, **kwargs)
        if self.router is not None:
            self.router = self.router.to(device)
        return self
    
    def add_expert(self, expert=None):
        """Adds new expert to ensemble"""
        if expert is None:
            # Create new expert with random feature set
            features_set = random.sample(range(self.input_size), self.input_size_sampled)
            
            # Create layer configuration for expert
            layer_configs_current = []
            for l in self.layer_configs:
                if self.exponential_layer_size:
                    value = int(np.random.exponential(scale=l))
                else:
                    value = l
                if value < 3:
                    value = 3
                layer_configs_current.append(value)
            
            # Create expert instance
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
                if self.lin_projection_bottleneck is not None:
                    cur_out_size = self.lin_projection_bottleneck
                else:
                    cur_out_size = self.out_size
                expert = ResNet(
                    self.input_size_sampled, 
                    cur_out_size, 
                    self.dropout_rate,
                    layer_configs_current,
                    False,
                    self.use_batchnorm,
                    self.use_activation,
                    self.activation,
                    bottleneck_sz=self.resnet_bottleneck_sz
                )
            
            # Save feature set
            expert.features = features_set
        else:
            # Use passed expert
            if not hasattr(expert, 'features'):
                expert.features = random.sample(range(self.input_size), self.input_size_sampled)
        
        # Add expert
        self.submodels.append(expert)
        
        # Rebuild router
        self._rebuild_router()
    
    def _rebuild_router(self):
        """Rebuilds router for current number of experts"""
        num_experts = len(self.submodels)
        
        if num_experts == 0:
            self.router = None
            return
        
        # Create router as ResNet
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
        #print(X)
        if len(self.submodels) == 0:
            return torch.zeros((X.size(0), self.out_size)), [] if self.by_submodels else None
        
        # Determine number of active experts
        active_k = self.top_k if self.training else self.inference_top_k
        active_k = min(active_k, len(self.submodels))
        
        # Split batch into subbatches
        outputs = []
        all_submodels_outputs = []
        
        for i in range(0, X.size(0), self.max_batch_size):
            #batch_slice = slice(i, min(i + self.max_batch_size, X.size(0)))
            X_batch = X[i: i + self.max_batch_size]
            
            # Calculate for subbatch
            batch_output, batch_submodels_outputs = self._forward_batch(X_batch, active_k)
            
            outputs.append(batch_output)
            if self.by_submodels:
                all_submodels_outputs.append(batch_submodels_outputs)
        
        # Combine results
        output = torch.cat(outputs, dim=0)
        
        if self.by_submodels:
            # Combine submodel outputs
            submodels_outputs = []
            for i in range(len(self.submodels)):
                expert_outputs = [sub[i] for sub in all_submodels_outputs]
                submodels_outputs.append(torch.cat(expert_outputs, dim=0))
            return output, submodels_outputs
        else:
            return output
    
    def _forward_batch(self, X, active_k):
        """Processes one batch (no more than max_batch_size)"""
        if not hasattr(self, "lin_projection_bottleneck"):
            self.lin_projection_bottleneck = None
        batch_size = X.size(0)
        shape = X.shape
        if len(X.shape) == 3:
            X = X.view(shape[0] * shape[1], shape[2])
        num_experts = len(self.submodels)
        
        # Calculate expert weights
        router_logits = self.router(X)
        weights = F.softmax(router_logits, dim=-1)
        
        
        # Select top-K experts
        if self.unlock_last_model:
            topk_weights, topk_indices = torch.topk(weights[:, :-1], k=active_k, dim=-1, sorted=False)
        else:
            topk_weights, topk_indices = torch.topk(weights, k=active_k, dim=-1, sorted=False)
        # if np.random.rand()<0.2:
        #     print(topk_indices)
        
        # Normalize weights
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
        
        # Calculate outputs of all experts if needed
        if self.by_submodels:
            all_expert_outputs = torch.zeros((batch_size, num_experts, self.out_size), device=X.device)
            for expert_idx, expert in enumerate(self.submodels):
                expert_input = X[:, expert.features]
                raw_out = expert(expert_input)
                if (self.lin_projection_bottleneck is not None) and (raw_out.shape[-1] == self.lin_projection_bottleneck):
                    all_expert_outputs[:, expert_idx] = self.linear_projector(raw_out)
                else:
                    all_expert_outputs[:, expert_idx] = raw_out
                del raw_out
        else:
            all_expert_outputs = None
        
        # Assemble total output
        output = torch.zeros((batch_size, self.out_size), device=X.device)
        
        # Group expert calls
        for expert_idx in range(num_experts):
            mask = (topk_indices == expert_idx).any(dim=1)
            
            if not mask.any():
                continue
                
            # Select examples for this expert
            expert_input = X[mask]
            expert = self.submodels[expert_idx]
            
            # Select needed features
            expert_input = expert_input[:, expert.features]
            
            # Calculate expert output
            if self.by_submodels:
                expert_output = all_expert_outputs[mask, expert_idx]
            else:
                expert_output = expert(expert_input)
                raw_out = expert(expert_input)
                if (self.lin_projection_bottleneck is not None) and (raw_out.shape[-1] == self.lin_projection_bottleneck):
                    expert_output = self.linear_projector(raw_out)
                    del raw_out
                else:
                    expert_output = raw_out
            
            # Get weights
            expert_weights = topk_weights[mask, (topk_indices[mask] == expert_idx).nonzero()[:, 1]]
            
            # Weighted sum
            output[mask] += expert_weights.unsqueeze(1) * expert_output

        if self.unlock_last_model:
            if self.by_submodels:
                expert_output = all_expert_outputs[:, -1]
            else:
                expert = self.submodels[-1]
                expert_input = X[:, expert.features]
                expert_output = expert(expert_input)
            output += expert_output
        
        # Apply sigmoid if needed
        if self.use_sigmoid_end:
            output = torch.sigmoid(output)

        #can forbid generation of, for example, stop tokens. Temporarily.
        if not hasattr(self, 'forbidden_tokens_list'):
            self.forbidden_tokens_list = []
            self.forbidden_tokens_counter = 0
        elif self.forbidden_tokens_counter > 0:
            self.forbidden_tokens_counter -= 1
            for t in self.forbidden_tokens_list:
               output[:, t] = torch.tensor(float('-inf'))
        nans = torch.isnan(output)
        if torch.any(nans):
            output[nans] = torch.tensor(float('-inf'))
        output[output<-1e6] = -1e6

        # Return submodel outputs if needed
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
        """Sets number of active experts for inference"""
        self.inference_top_k = k
