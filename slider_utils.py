import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class SlidingWindowLinear(nn.Module):
    def __init__(self, input_dim, output_dim, window_size):
        super(SlidingWindowLinear, self).__init__()
        self.window_size = window_size
        self.linear = nn.Linear(input_dim * window_size, output_dim)
        with torch.no_grad():
            self.linear.weight *= 0.0001
            self.linear.bias *= 0.0001
        print(self)
        
    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        # Паддинг для обеспечения контекста для первых window_size-1 элементов
        padded = torch.zeros(batch_size, seq_len + self.window_size - 1, emb_dim, device=x.device, dtype=x.dtype)
        padded[:, self.window_size - 1:, :] = x

        # Создаем 2D представление с окнами size (window_size)
        windows = padded.unfold(1, self.window_size, 1)  # Создаем окна
        
        # Вырежем размер до [batch_size, seq_len, window_size * emb_dim]
        windows = windows.contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, window_size * emb_dim]
        
        # Применяем линейный слой
        output = self.linear(windows)  # [batch_size, seq_len, output_dim]
        return output

class StackedSlidingWindow(nn.Module):
    def __init__(self, input_dim, output_dim, window_sizes, dropout_prob=0.05, activation=nn.LeakyReLU(), squeeze_dim=512, pool_size=12):
        super(StackedSlidingWindow, self).__init__()

        stride = 1
        padding = pool_size // 2  # Для сохранения длины последовательности
        self.padding = padding
        self.layers = nn.ModuleList()
        for window_size in window_sizes:
            self.layers.append(SlidingWindowLinear(squeeze_dim, squeeze_dim, window_size))
            self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=stride, padding=padding))
            self.layers.append(nn.LayerNorm(squeeze_dim))
            self.layers.append(activation)

        self.encoder = nn.Linear(input_dim, squeeze_dim)
        self.squeeze_dim = squeeze_dim
        self.out_linear =  nn.Linear(squeeze_dim + input_dim, input_dim)
        with torch.no_grad():
            self.out_linear.weight *= 0.00001
            self.out_linear.bias *= 0.00001

    def forward(self, x_input):
        x = self.encoder(x_input)
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], SlidingWindowLinear):
                x = self.layers[i](x)
            elif isinstance(self.layers[i], nn.Dropout):
                x = self.layers[i](x)  # Применяем Dropout
            elif isinstance(self.layers[i], nn.LayerNorm):
                x = self.layers[i](x)  # Применяем LayerNorm
            elif isinstance(self.layers[i], nn.MaxPool1d):
                x_padded = x[:, :-self.padding]
                x_padded = x_padded.permute(0, 2, 1)  # Применяем MaxPool1d
                x = self.layers[i](x_padded)
                x = x.permute(0, 2, 1)  # Вернем форму обратно для следующего слоя
                x = x[:, :-1, :]
                #западить нулями
                zero_padding = torch.zeros([x.shape[0], self.padding, x.shape[-1]], dtype=x.dtype, device=x.device)
                x = torch.hstack([zero_padding, x])
        x = self.out_linear(torch.dstack([x, x_input]))
        return x + x_input