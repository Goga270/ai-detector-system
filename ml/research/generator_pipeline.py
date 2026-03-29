import pandas as pd
import asyncio
import aiohttp
import random
import re
import os
import json
import logging
from datetime import datetime
from tqdm.asyncio import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from razdel import sentenize


# 0. НАСТРОЙКИ ЛОГИРОВАНИЯ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/generation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. НАСТРОЙКИ ОБЛАКА И НОД
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

LOCAL_NODES =[
    "http://localhost:8000/v1/chat/completions",
    "http://localhost:8001/v1/chat/completions"
]


YANDEX_MODELS =[
    f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-5-lite",
    f"gpt://{YANDEX_FOLDER_ID}/deepseek-v32"
    f"gpt://{YANDEX_FOLDER_ID}/qwen3-235b-a22b-fp8"
]

# 2. ПРОМПТЫ И СТРАТЕГИИ
EDIT_STRATEGIES = {
    "paraphrase": {
        "type": "full",
        "prompt": (
            "Ты — академический редактор. Сделай глубокий парафраз следующего текста, "
            "сохранив стиль и все ключевые смыслы. Не добавляй ничего от себя, не комментируй правки. "
            "СТРОГО ЗАПРЕЩЕНО использовать ИИ-штампы. "
            "Выдай только итоговый текст.\n\nТекст:\n{text}"
        ),
        "metadata": {"output_handling": "replace_full"}
    },
    "style_transfer_human": {
        "type": "full",
        "prompt": (
            "Перепиши текст так, чтобы он звучал как живое выступление увлечённого учёного. "
            "Добавь естественные рассуждения, убери канцеляризмы и излишнюю формальность. "
            "Сохрани научную достоверность. Не используй вводные слова-клише. "
            "Выдай только итоговый текст.\n\nИсходный текст:\n{text}"
        ),
        "metadata": {"output_handling": "replace_full"}
    },
    "partial_edit": {
        "type": "partial",
        "fraction": 0.4,
        "prompt": (
            "Ты редактор. Перепиши этот фрагмент текста другими словами, сохранив смысл и стиль. "
            "Выдай только переписанный фрагмент без кавычек и комментариев.\n\nФрагмент:\n{text}"
        ),
        "metadata": {"output_handling": "replace_fragment"}
    },
    "ai_from_middle": {
        "type": "continue",
        "fraction": 0.5,
        "prompt": (
            "Продолжи этот академический текст в том же стиле. Напиши продолжение, "
            "примерно равное по объёму исходному фрагменту. "
            "ВАЖНО: Не повторяй начало, просто продолжай с того места, где оборвано. "
            "Не пиши приветствий.\n\nНачало текста:\n{text}"
        ),
        "metadata": {"output_handling": "append"}
    }
}

JUDGE_PROMPTS = {
    "full": """
Ты строгий судья и эксперт по детекции ИИ.
Оригинал: {original}
Рерайт: {generated}
Оцени рерайт по шкале от 1 до 5 (где 5 - идеально сохраняет смысл и не содержит клише нейросетей, выглядит как текст человека).
Выдай СТРОГО JSON: {{"score": <число>, "feedback": "<критика, что исправить>"}}
""",
    "partial": """
Ты строгий судья. В человеческий текст была вставлена сгенерированная часть.
Оригинал: {original}
Текст со вставкой ИИ: {generated}
Оцени по шкале от 1 до 5 насколько бесшовным и естественным получился переход и не сломалась ли логика.
Выдай СТРОГО JSON: {{"score": <число>, "feedback": "<критика стыка>"}}
"""
}

# 3. БАЗОВЫЕ ФУНКЦИИ LLM И СУДЬИ
async def call_llm(session, system_prompt, user_prompt):
    """Маршрутизатор: Local vLLM -> Fallback Yandex"""
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": random.uniform(0.6, 0.9),
        "max_tokens": 1500
    }

    random.shuffle(LOCAL_NODES)
    for node_url in LOCAL_NODES:
        payload["model"] = "auto"
        try:
            async with session.post(node_url, json=payload, timeout=7.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'], f"Local_{node_url}"
        except (asyncio.TimeoutError, aiohttp.ClientError):
            continue

    y_model = random.choice(YANDEX_MODELS)
    yc_payload = {
        "modelUri": y_model,
        "completionOptions": {"stream": False, "temperature": payload["temperature"]},
        "messages": [{"role": "system", "text": system_prompt}, {"role": "user", "text": user_prompt}]
    }
    yc_headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": YANDEX_FOLDER_ID}

    try:
        async with session.post(YANDEX_URL, json=yc_payload, headers=yc_headers, ssl=False, timeout=10.0) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data['result']['alternatives'][0]['message']['text'], f"Yandex_{y_model}"
    except Exception as e:
        logger.error(f"Все LLM недоступны: {e}")
    return None, None


async def call_judge(session, original, generated, output_handling):
    """Вызов судьи для оценки текста"""
    j_type = "full" if output_handling == "replace_full" else "partial"
    prompt = JUDGE_PROMPTS[j_type].format(original=original, generated=generated)

    # Судья должен быть строгим, temperature=0.1
    response_text, _ = await call_llm(session, "Ты беспристрастный судья. Отвечай только JSON.", prompt)

    if not response_text:
        return 3, "Судья не ответил"

    try:
        clean_json = re.sub(r'```json|```', '', response_text).strip()
        result = json.loads(clean_json)
        return int(result.get("score", 3)), result.get("feedback", "")
    except Exception:
        return 3, "Ошибка парсинга JSON судьи"


# 4. АГЕНТСКИЙ ГЕНЕРАТОР С РЕФЛЕКСИЕЙ
async def generate_with_reflexion(session, original_text, chunk_id):
    """Цикл генерации, оценки и улучшения (макс 3 попытки)"""
    strategy_name = random.choice(list(EDIT_STRATEGIES.keys()))
    strategy = EDIT_STRATEGIES[strategy_name]
    handling = strategy["metadata"]["output_handling"]

    sentences = [s.text for s in sentenize(original_text)]
    if len(sentences) < 4 and handling != "replace_full":
        strategy_name = "paraphrase"
        strategy = EDIT_STRATEGIES["paraphrase"]
        handling = "replace_full"

    if handling == "replace_full":
        target_text = original_text
    elif handling == "replace_fragment":
        num_sents = max(1, int(len(sentences) * strategy["fraction"]))
        start_idx = random.randint(1, len(sentences) - num_sents - 1)
        target_text = " ".join(sentences[start_idx: start_idx + num_sents])
    elif handling == "append":
        mid = len(sentences) // 2
        target_text = " ".join(sentences[:mid])

    base_prompt = strategy["prompt"].format(text=target_text)
    current_prompt = base_prompt

    MAX_ATTEMPTS = 3
    best_text = None
    best_score = 0
    final_source = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # 1. Генерируем
        ai_part, source = await call_llm(session, "Ты научный писатель.", current_prompt)
        if not ai_part:
            break

        # 2. Собираем текст
        if handling == "replace_full":
            full_draft = ai_part
        elif handling == "replace_fragment":
            draft_sents = sentences.copy()
            draft_sents[start_idx: start_idx + num_sents] = [ai_part]
            full_draft = " ".join(draft_sents)
        elif handling == "append":
            full_draft = f"{target_text} {ai_part}"

        # 3. Судим
        score, feedback = await call_judge(session, original_text, full_draft, handling)

        # ЛОГИРУЕМ ПОПЫТКУ
        logger.info(
            f"[{chunk_id}] Попытка {attempt}/3 | Оценка: {score} | Стратегия: {strategy_name} | Фидбек: {feedback}")

        if score > best_score:
            best_score = score
            best_text = full_draft
            final_source = source

        # 4. Рефлексия
        if score >= 4:
            break
        else:
            current_prompt = f"{base_prompt}\n\nПредыдущая попытка была отклонена редактором. Критика: {feedback}\nПожалуйста, исправь ошибки и перепиши текст."

    if best_text:
        return {
            'doc_id': chunk_id.rsplit('_', 1)[0],
            'chunk_id': f"{chunk_id}_ai_{strategy_name}",
            'text': best_text,
            'label': 1,
            'source_model': final_source,
            'edit_strategy': strategy_name,
            'judge_score': best_score
        }
    return None


# 5. СЕМАНТИЧЕСКИЙ ЧАНКЕР (УВЕЛИЧЕННЫЙ)
class SemanticTextChunker:
    def __init__(self, model_name='cointegrated/rubert-tiny2', similarity_threshold=0.5, min_words=400, max_words=800):
        logger.info(f"📥[Semantic] Загрузка модели эмбеддингов {model_name}...")
        self.encoder = SentenceTransformer(model_name, device='cpu')
        self.similarity_threshold = similarity_threshold
        self.min_words = min_words
        self.max_words = max_words

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', str(text))
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        return text.strip()

    def chunk_text(self, text):
        text = self.clean_text(text)
        sentences =[s.text for s in sentenize(text)]
        if len(sentences) <= 2: return [" ".join(sentences)]

        embeddings = self.encoder.encode(sentences, show_progress_bar=False)
        similarities =[cosine_similarity([embeddings[i]],[embeddings[i+1]])[0][0] for i in range(len(embeddings)-1)]

        chunks, current_chunk = [], [sentences[0]]
        current_word_count = len(sentences[0].split())

        for i, sentence in enumerate(sentences[1:]):
            sim = similarities[i]
            word_count = len(sentence.split())

            is_topic_changed = sim < self.similarity_threshold and current_word_count >= self.min_words
            is_too_big = current_word_count + word_count > self.max_words

            if is_topic_changed or is_too_big:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_word_count = [sentence], word_count
            else:
                current_chunk.append(sentence)
                current_word_count += word_count

        if current_chunk: chunks.append(" ".join(current_chunk))
        return chunks

    def process_dataframe(self, df, text_col='text', id_col='article_id'):
        logger.info(f"🧠 [Semantic] Начинаем нарезку {len(df)} статей...")
        chunked_data =[]
        for _, row in df.iterrows():
            chunks = self.chunk_text(row[text_col])
            for i, chunk in enumerate(chunks):
                word_count = len(chunk.split())
                if word_count >= 20: # Слегка подняли порог от мусора
                    chunked_data.append({
                        'doc_id': row[id_col],
                        'chunk_id': f"{row[id_col]}_{i}",
                        'text': chunk,
                        'word_count': word_count,
                        'label': 0,
                        'source_model': 'Human',
                        'edit_strategy': 'original',
                        'judge_score': 5
                    })
        logger.info(f"✅ Готово! Получено {len(chunked_data)} человеческих чанков.")
        return pd.DataFrame(chunked_data)


# 6. ОРКЕСТРАТОР ПАЙПЛАЙНА
def save_chunk(chunk_data, filename="final_dataset.csv"):
    pd.DataFrame([chunk_data]).to_csv(filename, mode='a', index=False, header=not os.path.exists(filename))


async def run_pipeline(human_chunks_df, filename="final_dataset.csv"):
    processed_ids = set()
    if os.path.exists(filename):
        processed_ids = set(pd.read_csv(filename)['chunk_id'].unique())

    if not os.path.exists(filename):
        human_chunks_df.to_csv(filename, index=False)
        processed_ids.update(human_chunks_df['chunk_id'].unique())

    records = human_chunks_df.to_dict('records')
    processed_base_ids = set([pid.split('_ai_')[0] for pid in processed_ids if '_ai_' in pid])
    to_generate = [r for r in records if r['chunk_id'] not in processed_base_ids]

    logger.info(f"🚀 Нужно сгенерировать: {len(to_generate)} чанков.")

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(5)

        async def worker(row):
            async with sem:
                res = await generate_with_reflexion(session, row['text'], row['chunk_id'])
                if res:
                    save_chunk(res, filename)
                return res

        tasks = [worker(r) for r in to_generate]
        results = await tqdm(asyncio.gather(*tasks), total=len(tasks), desc="Генерация")

        count = sum(1 for r in results if r is not None)
        logger.info(f"🏁 Завершено. Успешно сгенерировано {count} чанков.")

# 7. ЗАПУСК
async def main():
    logger.info("Загрузка датасета...")
    df_raw = pd.read_csv("cyberleninka_articles_20260307_124155_cleaned.csv")

    chunker = SemanticTextChunker(min_words=500, max_words=800)
    human_chunks_df = chunker.process_dataframe(df_raw, text_col='text_clean', id_col='article_id')

    await run_pipeline(human_chunks_df)


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.run(main())
