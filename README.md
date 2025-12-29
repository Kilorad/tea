TEA - Tail Embedding Adapter

WHAT'S EVEN GOING ON HERE?

A very simple idea based on two premises:
1) LLMs mainly consist of sequential layers, and their LM head is lightweight.
2) We know how to create tabular models that converge much better, faster, and more reliably than transformers. Usually, such a tabular model is CatBoost, but we will use EResNetProb - this is a kind of analogue of boosting and random forest, but trainable via backpropagation, with similar pros and cons.

So, our pipeline:
1) Take an LLM.
2) Take a dataset.
3) Run inference on the LLM for a dataset batch, taking not the output tokens or even logits from the LLM, but the embeddings from the output layer.
4) Create an "embedding - correct token" table.
5) Take a training step for the tabular model on this table (boosting could be used, but it works poorly if there is a large set of output classes).
6) Insert this model instead of the LM-head in the LLM.

Analogue, advantages, and disadvantages.
This approach "competes" with LoRA.
TEA Pros:
- Much faster if you have a large dataset.
- Increases the size of the neural network, meaning it can make a less capable model more capable.
- If TEA has few submodels, it's quite fast at inference. If there are many, it's quite resistant to overfitting. So, to reiterate, adding 1 billion parameters to a model via transformer layers will take longer to execute than adding the same billion parameters via TEA's linear layers.
- Relatively easily adds new "knowledge" to the neural network - easier than LoRA. Meaning, in fewer hours.
TEA Cons:
- If the original model couldn't do something, a very large adapter will be needed to teach it that.
- During training, initially the generation quality will be like that of the original model, then it will drop, then it will rise higher. LoRA doesn't have this dip.
- A model with TEA runs slower than a model with LoRA (because there are more layers).

The main training script is `make_model_composed`. Composed - because previously I collected embedding-token pairs in a file and trained on them separately, and that was not composed.

Key hyperparameters and flags:
`start_train` - set to `True` if you're just starting to train the model, and `False` if a tail adapter checkpoint is already in the folder.

`learnable_linear_model` - alongside the resnets, you will also have the original lm_head. This can also be trained via this hyperparameter. If trained, the training process is generally much faster, but generation quality is less stable. Furthermore, if this is `True`, you cannot use conservativity (it will have to be zeroed).
`conservativity` - the higher it is, the more we are tied to how the original, base model would generate. This is a number from 0 to infinity, but in practice, setting it above 2 seems pointless. The higher the conservativity, the less chance of getting an unstable model. The lower the conservativity - the more "original" the model.
`composition_size` - the number of submodels. If 1, then we have one resnet; if many, then we have a whole random forest or boosting ensemble of them. The higher the `net_dropout_rate`, the more the structure resembles a random forest, i.e., less overfitting; the lower the `net_dropout_rate`, the more it resembles boosting, i.e., better accuracy.

Example checkpoint of a tail adapter for Llama 3.1 8B 4-bit:
https://disk.yandex.ru/d/P6cfejgLR0sWpg
How to attach to your model:
`model.lm_head = head`
Where `head` is what's in the checkpoint.

Example micro-datasets.
This is an RL dataset, meaning rewards are assigned: https://disk.yandex.ru/d/YkBhPEz32B8f5Q
And this is a non-RL and non-instruct dataset, meaning just text strings: https://disk.yandex.ru/d/yx3yAffIB01lVw

UPDATE FROM 23.03
I also added a slider and speculative generation. Slider allows TEA to take not one, but several embeddings as input.
But this logic is completely incompatible with the `generate` function in LLMs. Therefore, I wrote my own `generate` - it's slower by default, but you can generate several tokens at once, meaning in that case, it's actually faster. Example of running it:
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

Update from 20.04
Integrated an adapter that combines transformer layers and resnets. Transformer layers allow for creating more representative embeddings for our task. Details can be found in `tea_transformer_heavyhead.ipynb`.

Update from 27.05
Added the script `tea_full_caches.ipynb`. This is a caching scenario. You can disable training of the transformer adapter (set `['transformer_update_rate'] = 0`) after it has already been trained to some extent. Then enable the mode `cache_mode = 'only_cache'`. You will get caches with embedding-label pairs. Then you can set `cache_mode = 'train_from_cache'` - and then we get much faster training. The point here is that sometimes you need to experiment with different adapter architectures, but on the same dataset. And you need to conduct many such experiments quickly.
The mode `cache_mode = 'only_train'` is training as before.