from torch.utils.data import Dataset, DataLoader
import random
import pickle
from collections import deque, defaultdict

import pandas as pd
import torch


class InstructDatasetR(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_length=1000000, cut=None, seed=None, noise_config=None, ignore_rewards=False):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cut = cut
        self.ignore_rewards = ignore_rewards
        self.seed = seed
        self.noise_config = noise_config or {
            'input': {
                'skip_prob': 0.03,  # probability of substring skipping in input data
                'insert_prob': 0.03,  # probability of substring insertion in input data
                'swap_prob': 0.03,    # probability of substring swapping in input data
                'max_skip_length': 20, # maximum length of skipped substring
                'max_insert_length': 20, # maximum length of inserted substring
            },
            'target': {
                'skip_prob': 0.02,
                'insert_prob': 0.03,
                'swap_prob': 0.03,
                'max_skip_length': 12,
                'max_insert_length': 12,
            },
            'use_random_inserts': False,  # use random inserts or text fragments
            'min_length': 5,  # minimum string length after transformations
        }
        self.data = self.load_data()
        self.log_samples = deque(maxlen=45)

    def calculate_token_frequencies(self, sample_size=None):
        """Lightning-fast token frequency calculation via batch processing"""
    
        # Collect all labels into one list
        all_labels = [item[1] for item in self.data]
        
        # Apply sampling if needed
        if sample_size and sample_size < len(all_labels):
            all_labels = random.sample(all_labels, sample_size)
        
        # BATCH tokenization (main speed boost)
        batch_size = 10000
        token_counter = defaultdict(int)
        total_tokens = 0
        
        for i in range(0, len(all_labels), batch_size):
            batch = all_labels[i:i+batch_size]
            encoded = self.tokenizer.batch_encode_plus(
                batch,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False  # Speeds up by 15-20%
            )
            
            # Count tokens (ignoring padding)
            for seq in encoded['input_ids']:
                for token_id in seq:
                    if token_id != self.tokenizer.pad_token_id:
                        token_counter[token_id.item()] += 1
                        total_tokens += 1
        
        # Create DataFrame (similar to your code)
        freq_data = []
        for token_id, count in token_counter.items():
            try:
                token_str = self.tokenizer.decode(token_id)
                freq_data.append({
                    'token_id': token_id,
                    'token': repr(token_str),
                    'count': count,
                    'frequency': count / total_tokens,
                    'is_special': token_id in self.tokenizer.all_special_ids
                })
            except:
                freq_data.append({
                    'token_id': token_id,
                    'token': f'[INVALID_{token_id}]',
                    'count': count,
                    'frequency': count / total_tokens,
                    'is_special': True
                })
        
        df = pd.DataFrame(freq_data)
        return df.sort_values('count', ascending=False).reset_index(drop=True)
        
    def load_data(self):
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
        if self.seed is not None:
            random.seed(self.seed)
        for i in range(len(data) - 1, 0, -1):
            if ('<0>' in data[i][1]) or ('<r0>' in data[i][1]):
                del data[i]
            if self.ignore_rewards:
                for r_variant in [-2, -1, -0.5, 0]:
                    if len(data) > i and len(data[i]) > 1:
                        s = f"<r{r_variant}>"
                        if s in data[i][1]:
                            del data[i]
                            #rep += f'{s} in label'
                            break
                        s = f"<{r_variant}>"
                        if s in data[i][1]:
                            del data[i]
                            #rep += f'{s} in label'
                            break
                    else:
                        break
                
        data = random.sample(data, len(data))
        
        return data

    def __len__(self):
        return len(self.data)

    def _apply_noise(self, text, noise_params):
        """Applies noise to text according to parameters"""
        if not text or len(text) < self.noise_config['min_length']:
            return text
            
        text = list(text)  # work at character level for careful noising
        
        # Substring skipping
        if random.random() < noise_params['skip_prob'] and len(text) > self.noise_config['min_length']:
            skip_len = random.randint(1, min(noise_params['max_skip_length'], len(text) - self.noise_config['min_length']))
            skip_pos = random.randint(0, len(text) - skip_len)
            del text[skip_pos:skip_pos+skip_len]
        
        # Substring insertion
        if random.random() < noise_params['insert_prob']:
            insert_len = random.randint(1, noise_params['max_insert_length'])
            if self.noise_config['use_random_inserts']:
                # Insert random characters
                insert_text = [random.choice(' абвгдеёжзийклмнопрстуфхцчшщъыьэюяqwertyuiopasdfghjkl;[]zxcvbnm,.1234567890-=+_') for _ in range(insert_len)]
            else:
                # Insert substring from existing text
                if len(text) > insert_len:
                    start_pos = random.randint(0, len(text) - insert_len)
                    insert_text = text[start_pos:start_pos+insert_len]
                else:
                    insert_text = text[:insert_len]
            
            insert_pos = random.randint(0, len(text))
            text[insert_pos:insert_pos] = insert_text
        
        # Substring swapping
        if random.random() < noise_params['swap_prob'] and len(text) > 1:
            # Select two non-overlapping substrings of length 1 or 2
            max_swap_len = min(2, len(text) // 2)
            swap_len = random.randint(1, max_swap_len)
            
            if len(text) > 2 * swap_len:
                pos1 = random.randint(0, len(text) - 2 * swap_len)
                pos2 = random.randint(pos1 + swap_len, len(text) - swap_len)
                
                # Swap substrings
                substr1 = text[pos1:pos1+swap_len]
                substr2 = text[pos2:pos2+swap_len]
                text[pos1:pos1+swap_len] = substr2
                text[pos2:pos2+swap_len] = substr1

        return ''.join(text)

    def __getitem__(self, idx):
        
        self.log_samples.append(self.data[idx])
        parts = self.data[idx]
        text = parts[0]
        label = parts[1]
        parts = [parts[0], parts[1], 1]
        parts[-1] = 1
        
        # Extract multiplier r
        #rep = ''
        for r_variant in [-2, -1, -0.5, 0.5, 1, 2, 0]:
            s = f"<r{r_variant}>"
            if s in label:
                parts[-1] = r_variant
                label = label.replace(s, '')
                #rep += f'{s} in label'
                break
            s = f"<{r_variant}>"
            if s in label:
                parts[-1] = r_variant
                label = label.replace(s, '')
                #rep += f'{s} in label'
                break
        r = parts[-1]
        # if r == 0:
        #     del self.data[idx]
        #     idx += 1
        #     continue
        if text is None:
            text = label
        # Apply noise to input data
        if self.noise_config['input']['skip_prob'] > 0 or \
           self.noise_config['input']['insert_prob'] > 0 or \
           self.noise_config['input']['swap_prob'] > 0:
            text = self._apply_noise(text, self.noise_config['input'])
            
        # Apply noise to target data
        if self.noise_config['target']['skip_prob'] > 0 or \
           self.noise_config['target']['insert_prob'] > 0 or \
           self.noise_config['target']['swap_prob'] > 0:
            if len(label) > 30:
                label = self._apply_noise(label, self.noise_config['target'])
        
        # Check minimum length after transformations
        # if len(text) < self.noise_config['min_length'] or len(label) < self.noise_config['min_length']:
        #     continue
        
        # Encode text and label using tokenizer
        try:
            label_encoding = self.tokenizer.encode_plus(
                label,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                padding_side='left'
            )
            label_ids = label_encoding['input_ids']
            
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                padding_side='left'
                
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            
        except Exception as e:
            print(f"Error encoding sample: {e}")
            #continue
            
        #break
            
        return {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask,
            'labels': label_ids[0],
            'mult': torch.tensor(r)
        }


class InstructDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_length=512, seed=None):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        # Tokenize input and target separately
        input_encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length // 2,  # Reserve half for input
            padding=False,
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            label,
            max_length=self.max_seq_length // 2,  # Reserve half for target
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        # Extract input_ids and remove batch dimension
        input_ids = input_encoding['input_ids'].squeeze(0)
        target_ids = target_encoding['input_ids'].squeeze(0)

        # Concatenate input and target
        combined_ids = torch.cat([input_ids, target_ids], dim=0)

        # Ensure the combined sequence doesn't exceed max_seq_length
        if len(combined_ids) > self.max_seq_length:
            combined_ids = combined_ids[:self.max_seq_length]

        # Create attention mask (1s for all tokens, 0s for padding)
        attention_mask = torch.ones_like(combined_ids)
        if len(combined_ids) < self.max_seq_length:
            pad_length = self.max_seq_length - len(combined_ids)
            combined_ids = torch.cat([combined_ids, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])

        # Create labels: -100 for input tokens, target_ids for target tokens
        labels = torch.full_like(combined_ids, -100)
        target_length = len(target_ids)
        if len(combined_ids) >= len(input_ids) + target_length:
            labels[len(input_ids):len(input_ids) + target_length] = target_ids
        else:
            # Truncate target if necessary
            labels[len(input_ids):] = target_ids[:len(combined_ids) - len(input_ids)]

        return {
            'input_ids': combined_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }