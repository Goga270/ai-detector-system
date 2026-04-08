"""
llm_router.py
=============
Drop-in замена для call_llm() в generator_pipeline2.py.

Профили железа:
  LOCAL_AMD   — локальный vLLM, ROCm, 12 GB VRAM
                → Qwen2.5-7B-Instruct-AWQ (пик ~6.5 GB при bs=1)
  DATASPHERE  — YandexDataSphere GPU ≥ 30 GB
                → Qwen2.5-32B-Instruct-AWQ или Qwen2.5-14B-Instruct-GPTQ

Стратегия маршрутизации:
  1. Параллельный health-check всех LOCAL_NODES (timeout=3s).
  2. Запрос к первому живому узлу (timeout=45s для 7B, 90s для 32B).
  3. Если 0 живых узлов → Yandex Cloud fallback.
  4. Экспоненциальный backoff на Yandex при 429/5xx.

Использование:
  Замените функцию call_llm() в generator_pipeline2.py импортом:
    from llm_router import call_llm, RouterConfig, HardwareProfile
  Затем в main():
    RouterConfig.set_profile(HardwareProfile.LOCAL_AMD)  # или DATASPHERE
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import aiohttp
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# HARDWARE PROFILES
# ══════════════════════════════════════════════════════════════════

class HardwareProfile(str, Enum):
    LOCAL_AMD   = "local_amd"    # 12 GB VRAM, ROCm
    DATASPHERE  = "datasphere"   # 30+ GB VRAM, NVIDIA
    YANDEX_ONLY = "yandex_only"  # без локальных нод


@dataclass
class _ProfileSpec:
    nodes: list[str]
    model: str                       # модель для /v1/chat/completions
    node_timeout: float              # секунд на один запрос к ноде
    health_timeout: float = 3.0      # секунд на health-check
    max_tokens: int = 1500


_PROFILES: dict[HardwareProfile, _ProfileSpec] = {
    HardwareProfile.LOCAL_AMD: _ProfileSpec(
        nodes=["http://localhost:8001/v1/chat/completions"],
        # AWQ-квант: ~6.5 GB VRAM @ bs=1, ROCm-совместим
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        node_timeout=45.0,
    ),
    HardwareProfile.DATASPHERE: _ProfileSpec(
        # DataSphere экспортирует эндпоинт через SSH-туннель или прямой IP
        nodes=[
            os.getenv("DATASPHERE_NODE_URL", "http://localhost:8002/v1/chat/completions")
        ],
        # 32B AWQ: ~18-20 GB VRAM; если GPU < 24 GB — берите 14B GPTQ
        model=os.getenv("DATASPHERE_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
        node_timeout=90.0,
        max_tokens=2000,
    ),
    HardwareProfile.YANDEX_ONLY: _ProfileSpec(
        nodes=[],
        model="",
        node_timeout=0.0,
    ),
}

# ══════════════════════════════════════════════════════════════════
# GLOBAL CONFIG (мутируется один раз при старте из main())
# ══════════════════════════════════════════════════════════════════

class RouterConfig:
    _profile: HardwareProfile = HardwareProfile.YANDEX_ONLY
    _spec: _ProfileSpec = _PROFILES[HardwareProfile.YANDEX_ONLY]

    @classmethod
    def set_profile(cls, profile: HardwareProfile) -> None:
        cls._profile = profile
        cls._spec = _PROFILES[profile]
        logger.info(
            f"[Router] Profile set: {profile.value} | "
            f"nodes={cls._spec.nodes} | model={cls._spec.model}"
        )

    @classmethod
    def spec(cls) -> _ProfileSpec:
        return cls._spec


# ══════════════════════════════════════════════════════════════════
# YANDEX CONFIG
# ══════════════════════════════════════════════════════════════════

YANDEX_FOLDER_ID: str = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_API_KEY: str   = os.getenv("YANDEX_API_KEY", "")
YANDEX_URL            = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_MODELS         = [f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-5-lite"]
YANDEX_MODELS_GENERATOR = [f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-5.1"]
YANDEX_MODELS_JUDGE = [f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-5-lite"]
YANDEX_TIMEOUT        = 60.0
YANDEX_MAX_RETRIES    = 3
YANDEX_BASE_DELAY     = 2.0   # backoff base


# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

async def _is_node_alive(session: aiohttp.ClientSession, node_url: str, timeout: float) -> bool:
    """
    Пингует /health или делает минимальный probe-запрос.
    vLLM отдаёт /health 200 OK, если модель загружена.
    """
    health_url = node_url.replace("/v1/chat/completions", "/health")
    try:
        async with session.get(
            health_url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            ssl=False,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _get_live_nodes(
    session: aiohttp.ClientSession,
    nodes: list[str],
    health_timeout: float,
) -> list[str]:
    """Параллельная проверка всех нод, возвращает живые."""
    if not nodes:
        return []
    checks = await asyncio.gather(
        *[_is_node_alive(session, n, health_timeout) for n in nodes],
        return_exceptions=True,
    )
    live = [n for n, ok in zip(nodes, checks) if ok is True]
    if not live:
        logger.warning("[Router] All local nodes are DOWN → Yandex fallback")
    return live


# ══════════════════════════════════════════════════════════════════
# LOCAL vLLM CALL
# ══════════════════════════════════════════════════════════════════

class RateLimiter:
    """Максимум N запросов в секунду"""
    def __init__(self, rps: int = 15):
        self._sem = asyncio.Semaphore(rps)

    async def acquire(self):
        await self._sem.acquire()
        asyncio.get_event_loop().call_later(1.0, self._sem.release)

rate_limiter = RateLimiter(rps=15)

async def _call_local(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    live_nodes: list[str],
) -> Tuple[Optional[str], Optional[str]]:
    spec = RouterConfig.spec()
    nodes = live_nodes[:]
    random.shuffle(nodes)   # балансировка между нодами

    payload = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": spec.max_tokens,
    }

    for node_url in nodes:
        try:
            async with session.post(
                node_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=spec.node_timeout),
                ssl=False,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]
                    logger.debug(f"[Router] Local OK: {node_url} ({len(text)} chars)")
                    return text, f"Local_{spec.model}"
                else:
                    body = await resp.text()
                    logger.warning(f"[Router] Local {resp.status}: {body[:200]}")
        except asyncio.TimeoutError:
            logger.warning(f"[Router] Timeout on {node_url}")
        except aiohttp.ClientError as e:
            logger.warning(f"[Router] ClientError {node_url}: {e}")

    return None, None


# ══════════════════════════════════════════════════════════════════
# YANDEX CALL (с экспоненциальным backoff)
# ══════════════════════════════════════════════════════════════════

async def _call_yandex(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    role: str = "generator"
) -> Tuple[Optional[str], Optional[str]]:
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        logger.error("[Router] Yandex credentials missing.")
        return None, None
    models = YANDEX_MODELS_JUDGE if role == "judge" else YANDEX_MODELS_GENERATOR
    y_model = random.choice(models)
    payload = {
        "modelUri": y_model,
        "completionOptions": {"stream": False, "temperature": temperature},
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user",   "text": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
    }

    for attempt in range(1, YANDEX_MAX_RETRIES + 1):
        try:
            await rate_limiter.acquire()
            async with session.post(
                YANDEX_URL,
                json=payload,
                headers=headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=YANDEX_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["result"]["alternatives"][0]["message"]["text"]
                    logger.debug(f"[Router] Yandex OK ({len(text)} chars)")
                    return text, f"Yandex_{y_model}"

                # Rate limit / server error → backoff
                if resp.status in (429, 500, 502, 503):
                    delay = YANDEX_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.warning(
                        f"[Router] Yandex {resp.status}, retry {attempt}/{YANDEX_MAX_RETRIES} "
                        f"in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                # 4xx не retry
                body = await resp.text()
                logger.error(f"[Router] Yandex {resp.status}: {body[:300]}")
                return None, None

        except asyncio.TimeoutError:
            logger.warning(f"[Router] Yandex timeout attempt {attempt}/{YANDEX_MAX_RETRIES}")
            await asyncio.sleep(YANDEX_BASE_DELAY * attempt)
        except aiohttp.ClientError as e:
            logger.error(f"[Router] Yandex ClientError: {e}")
            return None, None

    logger.error("[Router] Yandex: all retries exhausted.")
    return None, None


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — замена call_llm() в generator_pipeline2.py
# ══════════════════════════════════════════════════════════════════

async def call_llm(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
    role: str = "generator"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Маршрутизатор: Local vLLM (health-checked) → Yandex Cloud fallback.

    Returns:
        (text, source_label) или (None, None) если все провалились.
    """
    if temperature is None:
        temperature = random.uniform(0.6, 0.9)

    spec = RouterConfig.spec()

    # Пробуем локальные ноды только если профиль их предполагает
    if spec.nodes:
        live_nodes = await _get_live_nodes(session, spec.nodes, spec.health_timeout)
        if live_nodes:
            text, source = await _call_local(
                session, system_prompt, user_prompt, temperature, live_nodes
            )
            if text:
                return text, source
        # Ноды заявлены, но все мертвы / запрос не удался → fallback логируем явно
        logger.warning("[Router] Local nodes failed → falling back to Yandex")

    return await _call_yandex(session, system_prompt, user_prompt, temperature, role)


# ══════════════════════════════════════════════════════════════════
# DOCKER SNIPPET (для справки, не исполняется)
# ══════════════════════════════════════════════════════════════════

_DOCKER_SNIPPETS = """
# ── LOCAL AMD (ROCm, WSL2) ───────────────────────────────────────
# docker run --rm -it \\
#   --device=/dev/kfd --device=/dev/dri \\
#   --security-opt seccomp=unconfined \\
#   --group-add video --ipc=host \\
#   -p 8001:8000 \\
#   -e HF_TOKEN=$HF_TOKEN \\
#   vllm/vllm-openai-rocm:latest \\
#   --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
#   --quantization awq \\
#   --dtype half \\
#   --gpu-memory-utilization 0.90 \\
#   --max-model-len 4096 \\
#   --trust-remote-code

# ── DATASPHERE (NVIDIA, ≥30 GB) ──────────────────────────────────
# docker run --rm -it \\
#   --gpus all \\
#   -p 8002:8000 \\
#   -e HF_TOKEN=$HF_TOKEN \\
#   vllm/vllm-openai:latest \\
#   --model Qwen/Qwen2.5-32B-Instruct-AWQ \\
#   --quantization awq \\
#   --dtype bfloat16 \\
#   --gpu-memory-utilization 0.92 \\
#   --max-model-len 8192 \\
#   --trust-remote-code
"""
