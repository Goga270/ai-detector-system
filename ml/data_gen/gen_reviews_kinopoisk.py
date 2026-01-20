from datasets import load_dataset
import pandas as pd
from openai import OpenAI
import random
import time
from tqdm import tqdm

OPENROUTER_API_KEY = ""
FREE_MODELS = [
    "deepseek/deepseek-v3.2",
]

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def generate_reviews_batch(
    movie_names: list,
    n_reviews_per_movie: int = 3,
    delay: float = 1.0,
    save_every: int = 100
) -> pd.DataFrame:
    lengths = ['short', 'medium']
    results = []

    total = len(movie_names) * n_reviews_per_movie

    print(f"Всего фильмов: {len(movie_names)}")
    print(f"Отзывов на фильм: {n_reviews_per_movie}")
    print(f"Всего будет сгенерировано: {total} отзывов")
    print(f"Примерное время: {total * delay / 60:.1f} минут\n")

    with tqdm(total=total, desc="Генерация") as pbar:
        for movie_name in movie_names:
            for i in range(n_reviews_per_movie):
                length = random.choice(lengths)
                review, model_used = generate_review(movie_name, length)

                results.append({
                    'movie_name': movie_name,
                    'date': None,
                    'content': review if review else '[ОШИБКА ГЕНЕРАЦИИ]',
                    'prompt_type': length,
                    'model': model_used,
                    'is_generated': True
                })

                pbar.update(1)

                if len(results) % save_every == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv('generated_reviews_backup.csv', index=False, encoding='utf-8-sig')

                time.sleep(delay)

    return pd.DataFrame(results)


dataset = load_dataset("blinoff/kinopoisk")
df = dataset['train'].to_pandas()
df = df[['movie_name', 'date', 'content']]
unique_movies = df['movie_name'].unique().tolist()

df_original = df[['movie_name', 'date', 'content']].copy()
df_original['prompt_type'] = None
df_original['model'] = None
df_original['is_generated'] = False

df_generated = generate_reviews_batch(
    movie_names=unique_movies,
    n_reviews_per_movie=3,
    delay=1.0,
    save_every=100
)
columns_order = ['movie_name', 'date', 'content', 'prompt_type', 'model', 'is_generated']

df_original = df_original[columns_order]
df_generated = df_generated[columns_order]
df_combined = pd.concat([df_original, df_generated], ignore_index=True)
df_combined.to_csv('combined_reviews.csv', index=False, encoding='utf-8-sig')
df_generated.to_csv('generated_only.csv', index=False, encoding='utf-8-sig')