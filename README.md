TEA - Tail Embedding Adapter

ЧТО ТУТ ВООБЩЕ ПРОИСХОДИТ?

Очень простая идея, основанная на двух предпосылках:
1) LLM состоят в основном из sequential слоёв, а LM head у них легковесная
2) Мы умеем создавать табличные модели, которые сходятся намного лучше, быстрее и надёжнее, чем трансформеры. Обычно такой табличной моделью является catboost, но у нас будет EResNetProb - это некий аналог бустинга и random forest, но обучаемый через backpropagation, и имеющий похожие плюсы и минусы.

Итого, наш пайплайн:
1) Взять LLM
2) Взять датасет
3) Проинференсить LLM на батче датасета, причём из LLM брать не токены на выходе, и даже не логиты, а эмбеддинги выходного слоя.
4) Сделать таблицу "эмбеддинг - правильный токен"
5) Сделать шаг обучения табличной модели на этой таблице (можно было бы и бустинг, но он плохо работает, если большое множество классов на выходе)
6) Воткнуть эту модель вместо LM-head в LLM


Аналог, преимущества и недостатки.
Данный подход "конкурирует" с LoRA.
Плюсы TEA:
- намного быстрее, если у вас большой датасет
- увеличивает размер нейронки, то есть позволяет сделать из менее ёмкой более ёмкую
- если в TEA мало субмоделей, то он довольно быстрый на инференсе. Если много, то довольно устойчивый к переобучению. То есть, ещё раз, если к модели добавить 1 миллиард параметров через трансформерные слои, то это будет исполняться дольше, чем если те же миллиард параметров добавить через линейные слои TEA.
- Относительно легко добавляет в нейронку новые "знания" - легче, чем LoRA. В смысле, за меньшее число часов
Минусы TEA:
- Если исходная модель чего-то не умела, то понадобится очень большой адаптер, чтобы научить её этому.
- По ходу обучения вначале у вас будет качество генерации, как у исходной модели, затем просядет, затем поднимется выше. У LoRA нет этой просадки.
- Модель с TEA исполняется дольше, чем модель с LoRA (потому что слоёв больше)



Основной скрипт обучения - make_model_composed. Composed - потому что раньше я собирал пары эмбеддинг-токен в файл, и обучал на них раздельно, и там было не composed.

Ключевые гиперпараметры и флаги:
start_train - проставьте в True, если только начинаете учить модель, и False, если в папке уже лежит чекпоинт tail adapter-а

learnable_linear_model - у вас в параллель с резнетами будет ещё и исходная lm_head. Так вот, её можно тоже обучать через этот гиперпараметр. Если обучаете, то процесс обучения идёт в целом намного быстрее, но качество генерации менее стабильно. Кроме того, если здесь True, то не получится использовать conservativity (его придётся занулить)
conservativity - чем больше, тем сильнее мы привязаны к тому, как бы генерила исходная, оригинальная модель. Это число от 0 до бесконечности, но на практике больше 2 его, кажется, нет смысла ставить. Чем выше консервативность, тем меньше шансов получить неустойчивую модель. Чем меньше консервативность - тем модель "оригинальнее"
composition_size - число субмоделей. Если 1, то у нас один резнет, если много, то у нас целый random forest или бустинг из них. Чем выше net dropout rate, тем больше конструкция похожа на random forest, то есть меньше переобучается, чем ниже net dropout rate, тем больше это похоже на бустинг, то есть лучше точность.

Пример чекпоинта tail adapter-а для Llama 3.1 8B 4-bit:
https://disk.yandex.ru/d/P6cfejgLR0sWpg
Как всобачить в свою модель: 
model.lm_head = head
Где head - это то, что в чекпоинте

Примеры микродатасетов.
Это RL-ный датасет, то есть проставлены реворды: https://disk.yandex.ru/d/YkBhPEz32B8f5Q
А это неRL-ный и не instruct датасет, то есть только строки текста: https://disk.yandex.ru/d/yx3yAffIB01lVw


АПДЕЙТ ОТ 23.03
Я ещё добавил slider и спекулятивную генерацию. Slider позволяет TEA принимать на вход не оди эмбеддинг, а несколько.
Но эта логика совершенно несовместима с функцией generate в LLM. Поэтому я написал свою generate - она по дефолту медленнее, зато можно генерить по несколько токенов за раз, то есть в таком случае она наоборот, быстрее. Пример запуска:
```
prompt = "Привет, как дела?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
# Generate
temp = 0.9
top_p=0.01
max_new_tokens = 150
repetition_penalty = 1.2
top_k = 10

t = pd.Timestamp.now()
generate_ids = generate_utils.generate_speculative(model, inputs.input_ids.to(device), 
             slider=None, heavy_lm_head=None,
             top_p=top_p, temperature=temp, max_new_tokens=max_new_tokens, 
             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, 
             do_sample=True, repetition_penalty=repetition_penalty, early_stopping=False, 
             tokenizer=tokenizer, stop_strings=None, top_k=top_k, 
             return_dict_in_generate=False, use_cache=True, estimation_rule='0.2')
print(pd.Timestamp.now() - t)
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(answer)
```
