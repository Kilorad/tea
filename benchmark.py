#!pip install rouge rouge-score nltk

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import re

def evaluate_translation(predicted, reference):
    # Нормализация
    pred_norm = normalize_answer(predicted)
    ref_norm = normalize_answer(reference)
    
    # ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_norm, ref_norm)[0]
    
    # BLEU
    bleu = sentence_bleu([ref_norm.split()], pred_norm.split())
    
    return {"rouge": rouge_scores, "bleu": bleu}
    

def validate_instruction(response, min_words=None, max_words=None, stop_token=None):
    # Обрезаем ответ до первого стоп-токена
    if stop_token:
        response = response.split(stop_token)[0].strip()
    
    # Проверка длины
    word_count = len(response.split())
    length_ok = True
    if min_words is not None and word_count < min_words:
        length_ok = False
    if max_words is not None and word_count > max_words:
        length_ok = False
    
    # Проверка наличия стоп-токена (если требуется)
    stop_ok = (stop_token is None) or (stop_token in response)
    
    return {
        "length_ok": length_ok,
        "stop_ok": stop_ok,
        "word_count": word_count
    }
def exact_match(predicted, reference):
    return normalize_answer(predicted) == normalize_answer(reference)

def list_match(predicted, reference_list):
    pred_items = set(normalize_answer(p).strip() for p in predicted.split(","))
    ref_items = set(normalize_answer(r) for r in reference_list)
    return {
        "precision": len(pred_items & ref_items) / len(pred_items) if pred_items else 0,
        "recall": len(pred_items & ref_items) / len(ref_items)
    }

def normalize_answer(text):
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9а-яё\s,]", "", text)

class LLMBenchmark:
    def __init__(self):
        self.tests = [
            # Тесты на фактологию (Heroes-3)
            {
                "type": "fact",
                "prompt": "К какому городу относится юнит Angel? Ответь одним словом на английском, затем поставь |. Ответ:",
                "reference": "castle",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Перечисли всех существ 7 уровня из замка Necropolis через запятую на английском. Поставь | в конце. Ответ:",
                "reference": ["bone dragon", "ghost dragon"],
                "format": "list"
            },
            {
                "type": "fact",
                "prompt": "Перечисли всех неулучшенных существ из замка Stronghold через запятую на английском. Поставь | в конце. Ответ:",
                "reference": ["goblin", "wolf rider", "orc", "ogre", "roc", "cyclops", "behemoth"],
                "format": "list"
            },
            {
                "type": "fact", 
                "prompt": "Какой юнит имеет самую высокую скорость в замке Tower? Выбери из: Arch Mage, Giant, Stone Gargoyle, Master Genie. Ответь одним названием на английском, в конце поставь |. Ответ:",
                "reference": "master genie",
                "format": "multiple_choice"
            },
            {
                "type": "fact",
                "prompt": "Какая самая часто выпадающая вторичная способность класса героя Knight? Ответь одним словом на русском, в конце поставь '|'. Ответ:",
                "reference": "лидерство",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Перечисли все замки базовой версии Heroes 3 через запятую на русском, в конце поставь '|'. Ответ:",
                "reference": ["замок", "оплот", "башня", "инферно", "некрополис", "подземелье", "цитадель", "крепость"],
                "format": "list"
            },
            {
                "type": "fact",
                "prompt": "Какой монстр в Doom 2 имеет красный цвет и сферическую форму? Ответь на русском, в конце поставь '|'. Ответ:",
                "reference": "какодемон",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Какой юнит из Doom применяет автоматическую пушку? Выбери из: Манкубус, Какодемон, Имп, Арахнотрон, Арч-Вайл. Ответь одним названием на русском, в конце поставь |. Ответ:",
                "reference": "арахнотрон",
                "format": "multiple_choice"
            },
            {
                "type": "fact",
                "prompt": "Какой монстр в Doom 2 имеет стреляет самонаводящимися боеприпасами? Ответь на русском, в конце поставь '|'. Ответ:",
                "reference": "ревенант",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Какое основное оружие у Пинки из Doom? Ответь на русском, в конце поставь '|'. Ответ:",
                "reference": "зубы",
                "format": "single"
            },
            {
                "type": "fact", 
                "prompt": "Какой юнит из Serious Sam First Encounter применяет ракеты? Выбери из: Красный Биомеханоид, Жёлтый Арахнид, Скелет Клира, Гарпия. Ответь одним названием на русском, в конце поставь |. Ответ:",
                "reference": "красный биомеханоид",
                "format": "multiple_choice"
            },
            {
                "type": "fact",
                "prompt": "Какой юнит в Serious Sam the First Encounter использует самоподрыв как основное оружие? Ответь через запятую на русском, в конце поставь '|'. Ответ:",
                "reference": ["безголовый камикадзе", "болотный прыгун"],
                "format": "list"
            },
            {
                "type": "fact",
                "prompt": "Какое оружие есть в Serious Sam the Second Encounter и нет в Serious Sam the First Encounter? Ответь через запятую на английском, в конце поставь '|'. Ответ:",
                "reference": ["serious bomb", "sniper rifle", "firethrower", "chainsaw"],
                "format": "list"
            },
            
            # Тесты на перевод
            *[{
                "type": "translation",
                "prompt": f"Переведи на русский: {en_text} ПЕРЕВОД (закончить знаком '|')",
                "reference": ru_text
            } for ru_text, en_text in [
                (
                    "Сегодня солнечный день. Птицы поют на деревьях. Дети играют в парке. Ветер легкий и теплый. Мы решили устроить пикник.",
                    "Today is a sunny day. Birds are singing in the trees. Children are playing in the park. The wind is light and warm. We decided to have a picnic."
                ),
                (
                    "Технологии искусственного интеллекта развиваются быстро. Они меняют многие отрасли промышленности. Однако возникают вопросы этики. Нужно регулировать эту сферу. Будущее покажет результаты.",
                    "Artificial intelligence technologies are developing rapidly. They are changing many industries. However, ethical issues arise. This area needs regulation. The future will show the results."
                ),
                (
                    "Вчера мы ходили в музей. Там была выставка современного искусства. Некоторые работы было трудно понять. Но цветовые сочетания впечатляли. Мы провели там три часа.",
                    "Yesterday we went to the museum. There was an exhibition of modern art. Some works were difficult to understand. But the color combinations were impressive. We spent three hours there."
                )
            ]]
        ]
        
    def run(self, model, tokenizer, device):
        log_file = 'bench_log.txt'
        with open(log_file, 'w', encoding="utf-8") as f:
            s = ''
            f.write(s)
            
        results = []
        for test in self.tests:
            #try:
            # Генерация ответа
            inputs = tokenizer(test["prompt"], return_tensors="pt").to(device)
            repetition_penalty = 1.2
            generate_ids = model.generate(
                inputs.input_ids.to(device),
                stop_strings=["|"],
                max_new_tokens=100,
                temperature=0.01,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                repetition_penalty=repetition_penalty,
                early_stopping=True,
                use_cache=True,
                num_beams=1,
                tokenizer=tokenizer
            )
            generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]
            response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("|")[0].strip()
            log_file = 'bench_log.txt'
            with open(log_file, 'a', encoding="utf-8") as f:
                s = f'{test} \nANSW: {response}\n'
                f.write(s)
            
            # Валидация
            if test["type"] == "fact":
                if test["format"] == "single":
                    score = int(normalize_answer(response) == normalize_answer(test["reference"]))
                elif test["format"] == "list":
                    pred_items = [x.strip() for x in normalize_answer(response).split(",")]
                    ref_items = [normalize_answer(x) for x in test["reference"]]
                    correct = len(set(pred_items) & set(ref_items))
                    score = {
                        "precision": correct / len(pred_items) if pred_items else 0,
                        "recall": correct / len(ref_items)
                    }
                elif test["format"] == "multiple_choice":
                    options = [normalize_answer(opt) for opt in test.get("options", [])]
                    score = int(normalize_answer(response) in options)
            elif test["type"] == "translation":
                rouge = Rouge()
                if len(response ) < 10:
                    response = "Empty answer"
                scores = rouge.get_scores(normalize_answer(response), normalize_answer(test["reference"]))[0]
                bleu = sentence_bleu([test["reference"].split()], response.split())
                score = {
                    "rouge_l": scores["rouge-l"]["f"],
                    "bleu": bleu
                }
            
            results.append((test["type"], score))
                
            # except Exception as e:
            #     print(f"Error in test: {test['prompt']}\n{str(e)}")
            #     results.append((test["type"], 0))
        
        return self._aggregate(results)
    
    def _aggregate(self, results):
        aggregated = {
            "fact": {"total": 0, "count": 0},
            "translation": {"rouge_l": 0, "bleu": 0, "count": 0}
        }
        
        for type_, score in results:
            if type_ == "fact":
                if isinstance(score, dict):
                    aggregated["fact"]["total"] += (score["precision"] + score["recall"])/2
                else:
                    aggregated["fact"]["total"] += score
                aggregated["fact"]["count"] += 1
            elif type_ == "translation":
                aggregated["translation"]["rouge_l"] += score["rouge_l"]
                aggregated["translation"]["bleu"] += score["bleu"]
                aggregated["translation"]["count"] += 1
                
        return {
            "fact_score": aggregated["fact"]["total"] / aggregated["fact"]["count"] if aggregated["fact"]["count"] else 0,
            "translation_rouge": aggregated["translation"]["rouge_l"] / aggregated["translation"]["count"] if aggregated["translation"]["count"] else 0,
            "translation_bleu": aggregated["translation"]["bleu"] / aggregated["translation"]["count"] if aggregated["translation"]["count"] else 0,
        }

class DummyModel:
    def generate(self, prompt, stop_words, max_tokens):
        # Заглушка для тестирования
        answers = {
            "К какому городу относится юнит Angel?": "castle |",
            "Перечисли всех существ 7 уровня из замка Necropolis": "dread knight, ghost dragon |",
            "Какой юнит имеет самую высокую скорость в замке Tower?": "Master Genie |",
            "Какая основная способность класса героя Knight?": "Логистика |",
            "Переведите на английский: Сегодня солнечный день": "Today is a sunny day."
        }
        for key in list(answers.keys()):
            if key in prompt:
                return answers[key]
        return "This is empty answer"
        

#benchmark = LLMBenchmark()
#results = benchmark.run(DummyModel())
#print(results)