import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path

import aiohttp
import asyncio
import random

from core.config import config
from core.exceptions import LLMError, LLMTimeoutError
from utils.logger import get_logger

LLM_CACHE_FILE = "data/llm_cache.json"
KNOWLEDGE_BASE_FILE = "data/llm_knowledge.json"
CACHE_TTL_SECONDS = 300


class LLMClient:
    def __init__(self):
        self.logger = get_logger("llm_client")
        self._api_key = config.llm["api_key"]
        self._api_url = config.llm["api_url"]
        self._model = config.llm["model"]
        self._enabled = config.llm["enabled"]
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Dict] = {}
        self._knowledge_base: List[Dict] = []
        self._load_cache()
        self._load_knowledge()

    def _cache_path(self) -> Path:
        return Path(LLM_CACHE_FILE)

    def _knowledge_path(self) -> Path:
        return Path(KNOWLEDGE_BASE_FILE)

    def _load_cache(self):
        try:
            p = self._cache_path()
            if p.exists():
                with open(p) as f:
                    self._cache = json.load(f)
                self.logger.info(f"Loaded {len(self._cache)} LLM cache entries")
        except Exception:
            self._cache = {}

    def _save_cache(self):
        try:
            p = self._cache_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def _load_knowledge(self):
        try:
            p = self._knowledge_path()
            if p.exists():
                with open(p) as f:
                    self._knowledge_base = json.load(f)
                self.logger.info(f"Loaded {len(self._knowledge_base)} knowledge entries")
        except Exception:
            self._knowledge_base = []

    def _save_knowledge(self):
        try:
            p = self._knowledge_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(self._knowledge_base[-1000:], f, indent=2)
        except Exception:
            pass

    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        entry = self._cache.get(cache_key)
        if entry:
            age = (datetime.now() - datetime.fromisoformat(entry["cached_at"])).total_seconds()
            if age < CACHE_TTL_SECONDS:
                return entry["result"]
        return None

    def _set_cache(self, cache_key: str, result: Dict):
        self._cache[cache_key] = {
            "result": result,
            "cached_at": datetime.now().isoformat(),
        }
        self._save_cache()

    def store_knowledge(self, symbol: str, cache_key: str, analysis: Dict, outcome: Optional[Dict] = None):
        self._knowledge_base.append({
            "symbol": symbol,
            "cache_key": cache_key,
            "analysis": analysis,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_knowledge()

    def get_historical_analysis(self, symbol: str, limit: int = 50) -> List[Dict]:
        return [
            e for e in self._knowledge_base
            if e.get("symbol") == symbol
        ][-limit:]

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
        timeout: float = 5.0,
        cache_key: Optional[str] = None,
    ) -> Optional[str]:
        if not self._enabled:
            return None

        if cache_key:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached.get("content")

        await self._ensure_session()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self._session.post(
                    f"{self._api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    if response.status == 429:
                        retry_after = 2 ** attempt + random.uniform(0, 1)
                        self.logger.warning(f"LLM rate limited (429), retrying in {retry_after:.1f}s (attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_after)
                        continue
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"LLM API error {response.status}: {error_text}")
                        return None

                    result = await response.json()
                    content = result["choices"][0]["message"]["content"].strip()

                    if cache_key:
                        self._set_cache(cache_key, {"content": content})

                    return content

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    self.logger.warning(f"LLM request timed out, retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    self.logger.warning("LLM request timed out after retries")
            except Exception as e:
                self.logger.warning(f"LLM request failed ({type(e).__name__}): {e}")
                return None
        return None

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._api_key)

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
        self._save_cache()
        self._save_knowledge()
