from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import openai
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseModel(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], **kwargs) -> str:
        raise NotImplementedError


class StandardOpenAIModel(BaseModel):
    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = openai.OpenAI(api_key=api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 12000, **_: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            max_tokens=max_tokens,
        )
        if not response.choices:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content or ""


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_id: str = "GPT4o",
        model_api_version: str = "2024-06-01",
        api_key: str | None = None,
    ):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = openai.AzureOpenAI(
            api_key="dummy",
            api_version=model_api_version,
            base_url="https://llm-api.amd.com",
            default_headers={"Ocp-Apim-Subscription-Key": api_key},
        )
        self.client.base_url = f"https://llm-api.amd.com/openai/deployments/{self.model_id}"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 1.0, max_tokens: int = 12000, **_: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            n=1,
            stream=False,
            max_tokens=min(max_tokens, 16000),
        )
        if not response.choices:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content or ""


class StandardClaudeModel(BaseModel):
    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 16000, **_: Any) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=min(max_tokens, 16000),
        )
        if not response.content:
            raise ValueError("No response content returned from the API.")
        return response.content[0].text


class ClaudeModel(BaseModel):
    def __init__(self, model_id: str = "claude-sonnet-4", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.server = "https://llm-api.amd.com/claude3"
        self.headers = {"Ocp-Apim-Subscription-Key": api_key}

    @retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 1.0, max_tokens: int = 16000, **_: Any) -> str:
        response = requests.post(
            url=f"{self.server}/{self.model_id}/chat/completions",
            json={
                "messages": messages,
                "temperature": temperature,
                "stream": False,
                "max_completion_tokens": min(max_tokens, 16000),
                "max_tokens": min(max_tokens, 16000),
                "presence_Penalty": 0,
                "frequency_Penalty": 0,
            },
            headers=self.headers,
            timeout=600,
        )
        if response.status_code != 200:
            raise ValueError(f"API returned status {response.status_code}: {response.text}")
        result = response.json()
        if "content" in result and result["content"]:
            return result["content"][0]["text"]
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        raise ValueError(f"Unexpected response format: {result}")


class GeminiModel(BaseModel):
    def __init__(self, model_id: str = "gemini-2.5-pro-preview-05-06", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.server = "https://llm-api.amd.com/vertex/gemini"
        self.headers = {"Ocp-Apim-Subscription-Key": api_key}

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 1.0, max_tokens: int = 30000, **_: Any) -> str:
        response = requests.post(
            url=f"{self.server}/{self.model_id}/chat",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_P": 0.95,
                "presence_Penalty": 0,
                "frequency_Penalty": 0,
            },
            headers=self.headers,
            timeout=600,
        )
        if response.status_code != 200:
            raise ValueError(f"API returned status {response.status_code}: {response.text}")
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]


def create_model_client(provider: str, model_id: str, api_key: str | None):
    resolved_api_key = (
        api_key
        or os.getenv("PY_HIP_KERNEL2KERNEL_API_KEY")
        or os.getenv("HIP2HIP_API_KEY")
        or os.getenv("TORCH2HIP_API_KEY")
        or os.getenv("TORCH_MODU2FUNC_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError(
            "An API key is required. Pass --api-key or set PY_HIP_KERNEL2KERNEL_API_KEY "
            "(HIP2HIP_API_KEY, TORCH2HIP_API_KEY, and TORCH_MODU2FUNC_API_KEY are also accepted)."
        )

    normalized = provider.strip().lower()
    if normalized == "openai":
        return OpenAIModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "standard-openai":
        return StandardOpenAIModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "claude":
        return ClaudeModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "standard-claude":
        return StandardClaudeModel(api_key=resolved_api_key, model_id=model_id)
    if normalized == "gemini":
        return GeminiModel(api_key=resolved_api_key, model_id=model_id)

    supported = ["openai", "standard-openai", "claude", "standard-claude", "gemini"]
    raise ValueError(f"Unsupported provider '{provider}'. Expected one of {supported}.")
