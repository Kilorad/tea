#!pip install rouge rouge-score nltk

import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def evaluate_translation(predicted, reference):
    # Нормализация
    pred_norm = normalize_answer(predicted)
    ref_norm = normalize_answer(reference)
    
    # ROUGE
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(pred_norm, ref_norm)[0]
        
        # BLEU
        bleu = sentence_bleu([ref_norm.split()], pred_norm.split())
    except Exception:
        rouge_scores = 0
        bleu = 0
    
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
                "prompt": "К какому городу из героев-3 относится юнит Angel? Ответь одним словом на английском, затем поставь |. Ответ:",
                "reference": "castle",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Перечисли всех существ 7 уровня из замка Necropolis через запятую на английском. Поставь | в конце. Пиши каждого в единственном числе. Ответ:",
                "reference": ["bone dragon", "ghost dragon"],
                "format": "list"
            },
            {
                "type": "fact",
                "prompt": "Перечисли всех неулучшенных существ из замка Stronghold в Heroes 3 через запятую на английском. Поставь | в конце. Пиши каждого в единственном числе. Ответ:",
                "reference": ["goblin", "wolf rider", "orc", "ogre", "roc", "cyclops", "behemoth"],
                "format": "list"
            },
            {
                "type": "fact",
                "prompt": "Как называется фракция (замок, город), из которой происходит Warlock в Героях меча и магии 3? Ответь на английском, одним словом. Поставь | в конце. Ответ:",
                "reference": "dungeon",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Какой из этих юнитов в Героях-3 является стреляющим? Василиск, монах, джинн, феникс? Ответь на русском, одним словом. Поставь | в конце. Ответ:",
                "reference": "монах",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Какой из этих юнитов в Героях-3 имеет ослепляющую атаку? Василиск, монах, джинн, феникс, единорог? Ответь на русском, одним словом. Поставь | в конце. Ответ:",
                "reference": "единорог",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Какой из этих юнитов относится к фракции Dungeon в Героях-3? Варианты: Скелет, Орк, Гарпия, Циклоп. Выбери вариант из списка. Поставь | в конце. Ответ:",
                "reference": "гарпия",
                "format": "single"
            },
            {
                "type": "fact", 
                "prompt": "Какой юнит имеет самую высокую скорость в замке Tower? Выбери из: Arch Mage, Giant, Stone Gargoyle, Master Genie. Ответь одним названием на английском, в конце поставь |. Ответ:",
                "reference": "master genie",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Какая самая часто выпадающая вторичная способность класса героя Knight? Ответь одним словом на русском, в конце поставь '|'. Ответ:",
                "reference": "лидерство",
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "У какого из этих героев Героях Меча и Магии 3 обычно выше всего spellpower? Knight, Wizard, Warlock, Cleric, Druid? Ответь одним словом на английском, в конце поставь '|'. Ответ:",
                "reference": "warlock",
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
                "format": "single"
            },
            {
                "type": "fact",
                "prompt": "Какой монстр из Doom сражается только в ближнем бою? Выбери из: Манкубус, Какодемон, Пинки, Имп, Арахнотрон, Арч-Вайл. Ответь одним названием на русском, в конце поставь |. Ответ:",
                "reference": "пинки",
                "format": "single"
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
                "prompt": "Какой из этих юнитов из Doom метает зелёные фаерболы? Revenant, Cacodemon, Hellknight, Imp, Mancubus. Назови только юнита на английском, в конце поставь '|'. Ответ:",
                "reference": "Hellknight",
                "format": "single"
            },
            {
                "type": "fact", 
                "prompt": "Какой юнит из Serious Sam First Encounter применяет ракеты? Выбери из: Красный Биомеханоид, Жёлтый Арахнид, Скелет Клира, Гарпия. Ответь одним названием на русском, в конце поставь |. Ответ:",
                "reference": "красный биомеханоид",
                "format": "single"
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
                ),
                (
                    '''Шэдоухарт осторожна и не склонна раскрывать информацию о себе, своих мотивах и реликвии, которую она носит. С особым недоверием она относится к гитьянки, таким как Лаэзель. Часто она кажется равнодушной к судьбе других людей, но не испытывает явного удовольствия от их страданий и не одобряет излишнюю жестокость и насилие. Обычно она одобряет поиск ненасильственных решений ситуаций, сострадательное отношение к животным и детям. Прежде всего Шэдоухарт ценит прагматизм.''',
                    "Shadowheart is cautious and reluctant to reveal information about herself, her motives, or the relic she carries. She is particularly distrustful of githyanki, such as Lae'zel. Often, she appears indifferent to the fate of others, though she does not take obvious pleasure in their suffering and disapproves of excessive cruelty and violence. She usually favors seeking nonviolent solutions to conflicts, showing compassion toward animals and children. Above all, Shadowheart values pragmatism."
                ),
                (
                    '''Ранний, или Древний, Сад (кон. IV — нач. III в. до н. э.) связан с деятельностью самого Эпикура и его ближайших учеников. В 32-летнем возрасте Эпикур организует свою школу сначала в Митилене, но вскоре бежав оттуда создаёт школу в Лампсаке (310 до н. э.), а затем, через 5 лет, переносит её в Афины (по другой версии, школа была основана в Афинах в 306 до н. э.). Школа находилась в саду философа, и по этой причине она получила название «Сад», а последователи Эпикура стали именоваться «философами из садов». В школу принимались женщины и рабы, причём, в отличие от пифагорейских общин, здесь отказываться от своего имущества не требовалось. На воротах школы была нанесена надпись: «Гость, тебе здесь будет хорошо; здесь удовольствие — высшее благо»''',
                    '''The Early (or Ancient) Garden
(late 4th – early 3rd century BCE)
The Early Garden is associated with the activities of Epicurus himself and his closest disciples. At the age of 32, Epicurus first established his school in Mytilene, but after fleeing from there, he soon founded another school in Lampsacus (310 BCE). Five years later, he moved it to Athens (though an alternative version suggests the school was founded directly in Athens in 306 BCE).
The school was located in the philosopher’s garden, which is why it became known as "The Garden" (Κῆπος, Kēpos). Its followers were called "the philosophers from the Gardens." Unlike the Pythagorean communities, the school admitted women and slaves, and there was no requirement to renounce personal property.
At the entrance, an inscription read:
"Stranger, here you will do well to stay; here our highest good is pleasure."'''
                ),
            ]]
        ]
        self.tests += [
            {
                "type": "reasoning",
                "prompt": {
                    "reasoning": (
                        "Петя старше Маши. Маша младше Васи. Кто самый старший?\n"
                        "Пожалуйста, рассмотри задачу шаг за шагом. Не давай окончательный ответ. Не более 40 слов. Рассуждение:"
                        "Закончи вывод символом @@"
                    ),
                    "answer": (
                        "Исходя из предыдущих рассуждений, кто самый старший? "
                        "Ответь строго в формате: [вариант]| где вариант - имя или 'недостаточно данных'. Ответ:"
                    )
                },
                "reference": "недостаточно данных",
                "format": "single",
                "max_tokens": {
                    "reasoning": 120,
                    "answer": 10
                }
            },
            {
                "type": "reasoning",
                "prompt": {
                    "reasoning": (
                        "В корзине 5 яблок. Добавили 3 груши и убрали 2 яблока. Сколько фруктов осталось?\n"
                        "Распиши решение по шагам. Не пиши окончательный ответ. Закончи вывод символом @@.  Не более 30 слов. Рассуждение:"
                    ),
                    "answer": (
                        "На основании вычислений, сколько фруктов осталось? "
                        "Ответь строго в формате: [число]|. Ответ:"
                    )
                },
                "reference": "6",
                "format": "single",
                "max_tokens": {
                    "reasoning": 120,
                    "answer": 10
                }
            },
            {
                "type": "reasoning",
                "prompt": {
                    "reasoning": (
                        "Вася сильнее Пети. Коля сильнее Васи. Петя сильнее Игоря. Игорь сильнее Павла. Кто сильнее: Игорь или Вася?\n"
                        "Распиши решение по шагам. Не пиши окончательный ответ. Закончи вывод символом @@. Не более 35 слов. Рассуждение:"
                    ),
                    "answer": (
                        "Кто сильнее?"
                        "Ответь строго в формате: [имя]|. Ответ:"
                    )
                },
                "reference": "вася",
                "format": "single",
                "max_tokens": {
                    "reasoning": 120,
                    "answer": 3
                }
            },
            {
                "type": "reasoning",
                "prompt": {
                    "reasoning": (
                        "Если все A являются B, а некоторые B являются C, верно ли что некоторые A являются C?"
                        "Распиши решение по шагам. Не пиши окончательный ответ. Закончи вывод символом @@. Не более 40 слов. Рассуждение:"
                    ),
                    "answer": (
                        "Верно ли что некоторые A являются C"
                        "Ответь 'да' или 'нет', затем поставь |. Ответ:"
                    )
                },
                "reference": "нет",
                "format": "single",
                "max_tokens": {
                    "reasoning": 120,
                    "answer": 3
                }
            },
            
        ]
        
    def run(self, model, tokenizer, device, trial_count=2):
        log_file = 'bench_log.txt'
        t = pd.Timestamp.now()
        with open(log_file, 'w', encoding="utf-8") as f:
            s = ''
            f.write(s)

        for trial in range(trial_count):
            results = []
            for test in self.tests:
                #try:
                # Генерация ответа
                if test["type"] != "reasoning":
                    if tokenizer is not None:
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
                        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("|")[0].strip().replace(".", "")
                        
                        if len(response)<=2:
                            generate_ids = model.generate(
                                inputs.input_ids.to(device),
                                stop_strings=["| "],
                                max_new_tokens=20,
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
                            response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                            if response[0] == '|':
                                response = response[1:]
                            elif response[1] == '|':
                                response = response[2:]
                            response = response.split("|")[0].strip().replace(".", "")
                        response = response.replace('|', '').replace('\n', '').strip()
                    else:
                        response = model.generate(test["prompt"], None, None).split("|")[0].strip().replace(".", "")
        
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
                    try:
                        scores = rouge.get_scores(normalize_answer(response), normalize_answer(test["reference"]))[0]
                        bleu = sentence_bleu([test["reference"].split()], response.split())
                        score = {
                            "rouge_l": scores["rouge-l"]["f"],
                            "bleu": bleu
                        }
                    except Exception:
                        score = {
                            "rouge_l": 0,
                            "bleu": 0
                        }
                elif test["type"] == "reasoning":
                    try:
                    #if 1:
                        # Этап 1: Генерация рассуждений
                        reasoning_prompt = test["prompt"]["reasoning"]
                        inputs = tokenizer(reasoning_prompt, return_tensors="pt").to(device)
                        reasoning_output = model.generate(
                            inputs.input_ids,
                            max_new_tokens=test["max_tokens"]["reasoning"],
                            temperature=0.2,
                            do_sample=True,
                            stop_strings=["@@"],
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            repetition_penalty=repetition_penalty,
                            use_cache=True,
                            tokenizer=tokenizer
                        )
                        reasoning_text = tokenizer.decode(reasoning_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                        reasoning_text = reasoning_text.split("@@")[0].strip()
        
                        # Этап 2: Генерация ответа
                        answer_prompt = f"{reasoning_prompt}\n{reasoning_text}\n\n{test['prompt']['answer']}"
                        inputs = tokenizer(answer_prompt, return_tensors="pt").to(device)
                        
                        answer_output = model.generate(
                            inputs.input_ids,
                            max_new_tokens=test["max_tokens"]["answer"],
                            temperature=0.0001,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            repetition_penalty=repetition_penalty,
                            use_cache=True,
                            tokenizer=tokenizer
                        )
                        answer_text = tokenizer.decode(answer_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                        final_answer = answer_text.split('|')[0]
        
                        # Валидация
                        #norm_answer = normalize_answer(final_answer)
                        score = int(str(test["reference"]).lower() in final_answer.lower())
        
                        # Логирование
                        with open(log_file, 'a', encoding="utf-8") as f:
                            f.write(f"### Test: {test['prompt']['reasoning'][:50]}...\n")
                            f.write(f"Reasoning: {reasoning_text}\n")
                            f.write(f"Final Answer: {final_answer} (Score: {score})\n\n")
        
                        results.append(("reasoning", score))
        
                    except Exception as e:
                        print(f"Error in reasoning test: {str(e)}")
                        results.append(("reasoning", 0))
                    
                with open(log_file, 'a', encoding="utf-8") as f:
                    s = f'{score} \n'
                    f.write(s)
                results.append((test["type"], score))
                
            # except Exception as e:
            #     print(f"Error in test: {test['prompt']}\n{str(e)}")
            #     results.append((test["type"], 0))
        t1 = pd.Timestamp.now()
        print('time', t1 - t)
        return self._aggregate(results)
    
    def _aggregate(self, results):
        
        aggregated = {
            "fact": {"total": 0, "count": 0},
            "translation": {"rouge_l": 0, "bleu": 0, "count": 0},
            "reasoning": {"total": 0, "count": 0}  # Добавлено
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
            # В цикле обработки результатов:
            elif type_ == "reasoning":
                aggregated["reasoning"]["total"] += score if isinstance(score, (int, float)) else 0
                aggregated["reasoning"]["count"] += 1.
        

        return {
            "fact_score": float(np.round(aggregated["fact"]["total"] / aggregated["fact"]["count"] if aggregated["fact"]["count"] else 0, 3)),
            "translation_rouge":  float(np.round(aggregated["translation"]["rouge_l"] / aggregated["translation"]["count"] if aggregated["translation"]["count"] else 0, 5)),
            "reasoning_score": float(np.round(aggregated["reasoning"]["total"] / aggregated["reasoning"]["count"], 3))
        }

class DummyModel:
    def generate(self, prompt, stop_words, max_tokens):
        # Заглушка для тестирования
        answers = {
            "К какому городу относится юнит Angel?": "castle |",
            "Перечисли всех существ 7 уровня из замка Necropolis": "bone dragon, ghost dragon |",
            "Какой монстр в Doom 2 имеет стреляет самонаводящимися боеприпасами": "Ревенант",
            "Какой юнит из Doom применяет автоматическую пушку?": "Арахнотрон",
            "Какой юнит имеет самую высокую скорость в замке Tower?": "Master Genie |",
            "Какая основная способность класса героя Knight?": "Лидерство |",
            "Переведите на английский: Сегодня солнечный день": "Today is a sunny day."
        }
        for key in list(answers.keys()):
            if key in prompt:
                return answers[key]
        return "This is empty answer"
        

#benchmark_cur = LLMBenchmark()
#results = benchmark_cur.run(DummyModel())
#print(results)