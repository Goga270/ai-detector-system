import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional

class YandexGPTClient:
    def __init__(self, api_key: str, folder_id: str, model_type: str = "yandexgpt"):
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.api_key = api_key
        self.folder_id = folder_id
        self.model_uri = f"gpt://{folder_id}/{model_type}"
        self.headers = {
            "Authorization": f"Api-Key {api_key}",
            "x-folder-id": folder_id,
            "Content-Type": "application/json"
        }

    async def generate(self, system_text: str, user_text: str, temperature: float = 0.3) -> Optional[str]:
        body = {
            "modelUri": self.model_uri,
            "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": 2000},
            "messages": [
                {"role": "system", "text": system_text},
                {"role": "user", "text": user_text}
            ]
        }

        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                try:
                    async with session.post(self.url, json=body, headers=self.headers) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return result['result']['alternatives'][0]['message']['text']
                        if resp.status == 429:
                            await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    logging.error(f"YGPT Error: {e}")
                    await asyncio.sleep(1)
        return None