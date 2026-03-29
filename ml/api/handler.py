import re
import pandas as pd
import joblib
from typing import Dict, Union, Tuple
import os
import sys
import json


### ГОША ###
### ПОДКРУТИ ДЛЯ СЕБЯ ВСЁ ###
### Я ПО ДРУГИМ ПАПКАМ РАСКИДАЛ ФАЙЛЫ, ЧТОБЫ КРАСИВО БЫЛО ###

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_root = os.path.dirname(os.path.dirname(current_dir))

model_path = os.path.join(ml_root, 'models', 'ai_text_classifier_pipeline.pkl')
stopwords_path = os.path.join(ml_root, 'models', 'russian_stopwords.pkl')

try:
    model = joblib.load(model_path)
    all_stopwords = joblib.load(stopwords_path)
except FileNotFoundError as e:
    # Если не нашли, выведем в консоль, где именно искали (поможет при отладке)
    print(f"Error: Model not found at {model_path}")
    raise e

#model = joblib.load('ai_text_classifier_pipeline.pkl')
#all_stopwords = joblib.load('russian_stopwords.pkl')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    text = text.lower().strip()

    return text

def analyze_text_features(text: str) -> Dict[str, float]:
    if not isinstance(text, str) or len(text) == 0:
        return {
            'punctuation_ratio': 0,
            'stopword_ratio': 0,
            'unique_word_ratio': 0,
            'uppercase_ratio': 0
        }

    words = text.lower().split()
    if len(words) == 0:
        return {
            'punctuation_ratio': 0,
            'stopword_ratio': 0,
            'unique_word_ratio': 0,
            'uppercase_ratio': 0
        }


    punctuation_count = sum(1 for char in text if char in '.,!?;:-')
    punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0
    stopword_count = sum(1 for word in words if word in all_stopwords)
    stopword_ratio = stopword_count / len(words) if len(words) > 0 else 0
    unique_word_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
    uppercase_count = sum(1 for char in text if char.isupper())
    uppercase_ratio = uppercase_count / len(text) if len(text) > 0 else 0

    return {
        'punctuation_ratio': punctuation_ratio,
        'stopword_ratio': stopword_ratio,
        'unique_word_ratio': unique_word_ratio,
        'uppercase_ratio': uppercase_ratio
    }

def predict_text_probability(text: str, return_details: bool = False) -> Union[float, Tuple[float, Dict]]:
    cleaned_text = clean_text(text)

    char_count = len(text)
    word_count = len(text.split())

    features = analyze_text_features(text)

    input_data = pd.DataFrame([{
        'content_clean': cleaned_text,
        'char_count': char_count,
        'word_count': word_count,
        'punctuation_ratio': features['punctuation_ratio'],
        'stopword_ratio': features['stopword_ratio'],
        'unique_word_ratio': features['unique_word_ratio']
    }])

    try:
        probability = model.predict_proba(input_data)[0, 1] * 100

        if return_details:
            confidence_level = 'High' if abs(probability - 50) > 30 else \
                              'Medium' if abs(probability - 50) > 15 else 'Low'

            details = {
                'ai_probability': round(probability, 2),
                'human_probability': round(100 - probability, 2),
                'prediction': 'AI Generated' if probability > 50 else 'Human',
                'confidence': confidence_level,
                'text_length_chars': char_count,
                'text_length_words': word_count,
                'features': {
                    'punctuation_ratio': round(features['punctuation_ratio'], 4),
                    'stopword_ratio': round(features['stopword_ratio'], 4),
                    'unique_word_ratio': round(features['unique_word_ratio'], 4)
                }
            }
            return probability, details
        else:
            return probability

    except Exception as e:
        if return_details:
            return 50.0, {'error': str(e), 'prediction': 'Unknown'}
        else:
            return 50.0


def ai_detection_api(text: str) -> Dict:
    try:
        prob, details = predict_text_probability(text, return_details=True)
        response = {
            'success': True,
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'analysis': {
                'ai_probability': details['ai_probability'],
                'human_probability': details['human_probability'],
                'prediction': details['prediction'],
                'confidence': details['confidence'],
                'text_length': {
                    'characters': details['text_length_chars'],
                    'words': details['text_length_words']
                },
                'features': details['features']
            },
            'interpretation': {
                'is_likely_ai': details['prediction'] == 'AI Generated',
                'confidence_score': 0.9 if details['confidence'] == 'High' else
                                   0.7 if details['confidence'] == 'Medium' else 0.5
            }
        }

        return response

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'text_preview': text[:100] + '...' if len(text) > 100 else text
        }


if __name__ == "__main__":
    #test_text = "Фильм оставляет смешанные чувства. С одной стороны, великолепная операторская работа..."
    #probability = predict_text_probability(test_text)
    #print(f"Вероятность ИИ: {probability:.2f}%")

    #prob, details = predict_text_probability(test_text, return_details=True)
    #print(f"Детали анализа:")
    #for key, value in details.items():
        #print(f"  {key}: {value}")

    #api_response = ai_detection_api(test_text)
    #print(f"API Response: {api_response}")

    # Временно пока делаем онли фронт из фронта запрос
    if len(sys.argv) > 1:
            input_text = sys.argv[1]
            result = ai_detection_api(input_text)
            print(json.dumps(result, ensure_ascii=False))



###     это для FASTAPI!!!      ###
###     это для FASTAPI!!!      ###
###     это для FASTAPI!!!      ###
###     это для FASTAPI!!!      ###

#     if __name__ == "__main__":
#         test_text = "Фильм оставляет смешанные чувства. С одной стороны, великолепная операторская работа..."


#         probability = predict_text_probability(test_text)
#         print(f"Вероятность ИИ: {probability:.2f}%")

#         prob, details = predict_text_probability(test_text, return_details=True)
#         print(f"Детали анализа:")
#         for key, value in details.items():
#             print(f"  {key}: {value}")


#         api_response = ai_detection_api(test_text)
#         print(f"API Response: {api_response}")
