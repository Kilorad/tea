from torch.utils.data import Dataset, DataLoader
import random
import pickle
from collections import deque
import torch

class InstructDatasetR(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_length=1000000, cut=None, seed=None):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cut = cut
        self.seed = seed
        self.data = self.load_data()
        self.log_samples = deque(maxlen=45)
        

    def load_data(self):
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
        if self.seed is not None:
            random.seed(self.seed)
        data = random.sample(data, len(data))
        
        # if self.cut is None:
        #     data = [data_cur for data_cur in data if len(data_cur[1])>5]
        # else:
        #     data = [data_cur for data_cur in data[:self.cut] if self.cut>5]
        # for i in range(100):
        #     print('***', data[i])
        #     try:
        #         if (not 'Алис' in data[i][0]) and (not 'Элеон"' in data[i][0]) and (not 'Софи"' in data[i][0]) and (not 'Сэм"' in data[i][0]) (not 'Алис' in data[i][1]) and (not 'Элеон"' in data[i][1]) and (not 'Софи"' in data[i][1]) and (not 'Сэм"' in data[i][1]):
        #             print('***', data[i])
        #     except Exception:
        #         pass
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while 1:
            self.log_samples.append(self.data[idx])
            parts = self.data[idx]
            text = parts[0]
            label = parts[1]
            parts = [parts[0], parts[1], 1]
            for r_variant in [-2,-1,-0.5,0.5,1,2]:
                s = f"<r{r_variant}>"
                if s in label:
                    parts[-1] = r_variant
                    label = label.replace(s, '')
                    break
                s = f"<{r_variant}>"
                if s in label:
                    parts[-1] = r_variant
                    label = label.replace(s, '')
                    break
            r = parts[-1]
            if r == 0:
                continue
            if text is None:
                text = label
            
            # Кодируем текст и метку с помощью tokenizer
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # Кодируем метку
            label_encoding = self.tokenizer.encode_plus(
                label,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            label_ids = label_encoding['input_ids']
            break
        return {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask,
            'labels': label_ids[0],
            'mult': torch.tensor(r)
        }