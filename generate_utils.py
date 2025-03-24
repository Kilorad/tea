import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import transformers

#будем генерить не 1, а несколько токенов за раз. Потом проверять.
#логика спекулятивного кодирования
#генерим не один логит, а несколько (блок). Все их превращаем в токены. Надо бы сделать отдельно функцию logit2token
#затем делаем следующий forward - но так, чтобы видеть скоры по только что сгенерённым токенам (кроме 1-го). Если один из токенов нас не устраивает - заменяем его на тот, что получается из логита
#соответственно, нужна функция, оценивающая, устраивает ли нас токен с учётом логита.
#это будет та же функция, что logit->token

#ещё деталь реализации: генерация следующего токена внутри цикла - это развилка, if. 
#Если в буфере есть что-либо - выдаём токен из буфера. Если нет - проверяем последние токены, насколько они соответствуют логитам, и генерим новые, записываем их в буфер
def logit2token(logits, top_p=0.9, temperature=1.0, do_sample=True, top_k=50, repetition_penalty=1.0, generated_sequence=None, estimate_token=None, estimation_rule=None):
    '''estimation_rule - how to check if generated tokens are good. Options:
    'precise' (tokens, made by .forward should be eqivalent to tokens, made by normal generate)
    'acceptable' (token is not forbidden by top_k, top_p, if normal generate started)
    'mean' (token has probability more than average, among non-forbidden by top_k, top_p)
    '0.3' (token has probability more than 30% quantile, among non-forbidden by top_k, top_p)
    '''
    lshape = logits.shape
    if len(lshape) == 3:
        logits = logits.reshape([lshape[0] * lshape[1], lshape[2]])
        if estimate_token is not None:
            estimate_token = estimate_token.reshape([lshape[0] * lshape[1]])

    estimated_token_good = torch.ones(logits.shape[0], device=logits.device)
    # Применяем temperature
    logits = logits/ temperature

    if repetition_penalty != 1.0:
        #перебор по строкам в батче, не бойся асимптотики, всё равно ты вряд ли будешь генерить больше одной строки за раз
        for i in range(generated_sequence.size(0)):
            # Собираем уникальные токены, кроме последнего
            unique_tokens = generated_sequence[i, :].unique()  # Извлекаем уникальные токены
            logits[i * lshape[1] : (i + 1) * lshape[1], unique_tokens] /= repetition_penalty  # Применяем штраф

        
        # Применяем Top-k и Top-p
        if do_sample:
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.zeros_like(logits).to(logits.device)
            filtered_logits.scatter_(1, top_k_indices, top_k_logits)
            logits = filtered_logits
            
            # Top-p sampling (nucleus sampling)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p

            # Необходимый (и не нужный) токены
            for i in range(sorted_indices_to_remove.size(0)):
                sorted_indices_to_remove[i, 1:] = sorted_indices_to_remove[i, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0  # Мы никогда не отсекаем первый токен

            # Убираем конечные токены
            
            for i in range(logits.shape[0]):
                indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = float('-inf')
                
                if estimation_rule == 'acceptable':
                    if not (logits[i, estimate_token[i]] > float('-inf')):
                        estimated_token_good[i] = 0

                if estimation_rule == 'mean':
                    idx = logits[i] > float('-inf')
                    estimated_token_good[i] = (logits[i,estimate_token[i]] > torch.mean(logits[i, idx])).to(torch.int16)
                    
                if estimation_rule == '0.3':
                    idx = logits[i] > float('-inf')
                    estimated_token_good[i] = (logits[i,estimate_token[i]] > torch.quantile(logits[i, idx].to(torch.float32), 0.3)).to(torch.int16)
            
            # Нормализуем вероятности
            probs = F.softmax(logits, dim=-1)

            # Сэмплинг из распределения
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Если не подавать выборку, берем argmax
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        
        if ((estimation_rule in ['acceptable', 'mean']) and (not do_sample)) or (estimation_rule == 'precise'):
            for i in range(logits.shape[0]):
                if next_token[i] != estimate_token[i]:
                    estimated_token_good[i] = 0

        if len(lshape) == 3:
            next_token = next_token.reshape([lshape[0], lshape[1]])
            if estimate_token is not None:
                estimated_token_good = estimated_token_good.reshape([lshape[0], lshape[1]])
                estimate_token = estimate_token.reshape([lshape[0], lshape[1]])
        return next_token, estimated_token_good


def generate_speculative(model, input_ids, heavy_lm_head=None, slider=None, max_new_tokens=50, top_p=0.9,top_k=50, temperature=1.0,  
             pad_token_id=None, eos_token_id=None, bos_token_id=None, 
             do_sample=True, repetition_penalty=1.0, early_stopping=False, 
             tokenizer=None, stop_strings=None, 
             return_dict_in_generate=False, use_cache=True, slider_window=100, 
             autoregression_step=3, filler=None, estimation_rule='mean', debug=False):
    '''speculative generation
    2 main ideas:
    1) You can change everything by your sly hands. This is much simplier than in normal .generate.
    2) You can process speculative generation. Call .forward method, make autoregression_step new logits, 
    convert them to tokens. Ckeck them by next .forward. Cancel if bad. Theoretically, LLM can make more than one correct token at a time, so it can generate text faster.
    
    model - LLM
    input_ids - tokens tensor, such as in normal generate method
    heavy_lm_head - optional. Tail embedding adapter, embedding on input, logit on output. Should be halved (in fp16)
    slider - optional. Aggregates embeddings before tail embedding adapter. 
    Special neural network (or something another with .forward method) to make single embedding for LM-head from numerous embeddings. Should be halved
    slider_window - width of your slider. How many embeddings it should process. Use 1 if no slider
    autoregression_step - how many tokens should be generated in parallel
    filler - token, that masks future. You can train model with special masking token. If not, try to use pad_token_id
    estimation_rule - how to check if generated tokens are good. Options:
    'precise' (tokens, made by .forward should be eqivalent to tokens, made by normal generate)
    'acceptable' (token is not forbidden by top_k, top_p, if normal generate started)
    'mean' (token has probability more than average, among non-forbidden by top_k, top_p)
    '0.3' (token has probability more than 30% quantile, among non-forbidden by top_k, top_p)
    '''
    #estimation_rule = 'precise' (точное совпадение) или 'acceptable' (не нулевая вероятность) или 'mean' (> среднего)
    if filler is None:
        #логика спекулятивного кодирования. Здесь пишем, чем заполняем конец строки для генерации более чем одного токена
        try:
            filler = pad_token_id
        except Exception:
            assert False, "You should provide something as filler. Default value of filler is pad_token_id. You should provide at least one of them."

    buffer_tokens = []
    with torch.no_grad():
        #slider_window - сколько эмбеддингов подать в slider. Вообще-то зависит от устройства slider, но я не знаю, как прокинуть его размер окна, так что пусть будет гиперпараметр
    
        # Стартовые данные
        generated_sequence = input_ids.clone()
        # Если задан bs_token_id, добавляем его в начало
        if len(generated_sequence.shape) == 1:
            generated_sequence = torch.stack([generated_sequence])
        if bos_token_id is not None:
            generated_sequence = torch.cat([torch.tensor([[bos_token_id]], device=input_ids.device), generated_sequence], dim=1)
    
        # Генерируем токены
        first_iter = True
        #for _ in range(max_new_tokens):
        factial_iterations_cnt = 0
        debug_report = {'iterations':0,'len_generated':0}
        while 1:
            if len(buffer_tokens) == 0:
                factial_iterations_cnt += 1
                #в буфере ничего нет - генерим!
                if autoregression_step > 1:
                    generated_sequence_w_paddings = torch.hstack([generated_sequence, torch.zeros([generated_sequence.shape[0], autoregression_step - 1], device=generated_sequence.device, dtype=torch.int64) + filler])
                else:
                    generated_sequence_w_paddings = torch.hstack([generated_sequence])
                outp = model.forward(generated_sequence_w_paddings, output_hidden_states=True, return_dict=True, use_cache=use_cache, output_scores=False, output_logits=False, past_key_values=None)
                embeddings = outp['hidden_states'][-1].detach()
        
                # Применяем нестандартный LM head
                #Slider учитывает нескольких последних эмбеддингов. Принимает их на вход, выдаёт некую оконную функцию
                if slider is not None:
                    embeddings_af_slider = slider.forward(embeddings[:, -slider_window - 2 * autoregression_step:])
                else:
                    embeddings_af_slider = embeddings
                #сама тяжёлая LM-head
                if heavy_lm_head is not None:
                    logits = heavy_lm_head(embeddings_af_slider[:, -2 * autoregression_step:, :])
                else:
                    logits = model.lm_head(embeddings_af_slider[:, -2 * autoregression_step:, :])
                logits_future = logits[:, -autoregression_step:]
                if not first_iter:
                    logits_past = logits[:, -autoregression_step * 2 + 1: -autoregression_step]

                #del logits
                next_tokens, _ = logit2token(logits_future, top_p=top_p, temperature=temperature, do_sample=do_sample, top_k=top_k, repetition_penalty=repetition_penalty, generated_sequence=generated_sequence, estimate_token=None, estimation_rule=None)
                buffer_tokens = next_tokens
                #проверить, правда ли недавно сгенерённые токены хороши
                if not first_iter:
                    position_start_check = max(generated_sequence.shape[1] - autoregression_step + 1, input_ids.shape[1])
                    if position_start_check > input_ids.shape[1]:
                        tokens_past_fixed, estimated_token_good = logit2token(logits_past, top_p=top_p, temperature=temperature, do_sample=do_sample, top_k=top_k, repetition_penalty=repetition_penalty, generated_sequence=generated_sequence[:, :], estimate_token=generated_sequence[:, position_start_check:], estimation_rule=estimation_rule)
                        estimated_token_good_bycolumns = torch.all(estimated_token_good, axis=0)
                        if torch.any(~estimated_token_good_bycolumns):
                            #отбросить токены, которые "не good". У нас в разных строках может быть по-разному, мы режем по самой короткой строке
                            idx_bad = torch.where(~estimated_token_good_bycolumns)[0][0]
                            offset = autoregression_step - idx_bad - 1
                            generated_sequence = generated_sequence[:, :-offset]
                            debug_report['len_generated'] -= offset.item()
                            #очистить буфер и поместить в него токены, полученные из проверочных логитов для прошлого
                            buffer_tokens = torch.hstack([tokens_past_fixed, next_tokens[:, :1]])
                #пора ли прекращать генерить
                if (generated_sequence.shape[1] >= max_new_tokens + input_ids.shape[1]):
                    break
                # Проверка на EOS токен
                if eos_token_id is not None and eos_token_id in generated_sequence:
                    break
                # Проверка на стоп-строки
                if stop_strings is not None:
                    generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
                    if any(stop_str in generated_text for stop_str in stop_strings):
                        break
            else:
                debug_report['iterations'] += 1
                #в буфере что-то есть - берём!
                # Добавляем новые токены к сгенерированной последовательности
                generated_sequence = torch.cat([generated_sequence, buffer_tokens], dim=1)
                debug_report['len_generated'] += buffer_tokens.shape[1]
                buffer_tokens = []
    
            first_iter = False
    generated_sequence = generated_sequence[:, input_ids.shape[1]:]
    if debug:
        debug_report['avg_tokens_generated_count'] = (debug_report['len_generated'] + 0.)/debug_report['iterations']
        print(debug_report)
    # Возвращаем сгенерированную последовательность
    if return_dict_in_generate:
        return {"generated_sequence": generated_sequence, "scores": logits}
    else:
        return generated_sequence