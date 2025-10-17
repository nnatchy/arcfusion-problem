"""
OpenAI LLM client wrapper for GPT-4o-mini.
Provides a clean interface for LLM calls with error handling and logging.
"""

from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from loguru import logger
from src.config import config


class LLMClient:
    """Async OpenAI LLM client wrapper"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.llm.model
        self.temperature = config.llm.temperature
        logger.info(f"✅ LLM client initialized (model={self.model})")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate completion from OpenAI.

        Args:
            system_prompt: System instruction
            user_prompt: User query/instruction
            temperature: Override default temperature
            response_format: Optional response format (e.g., {"type": "json_object"})
            max_tokens: Optional max tokens override

        Returns:
            Generated text response
        """
        temperature = temperature if temperature is not None else self.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            logger.debug(f"LLM call: model={self.model}, temp={temperature}")

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }

            if response_format:
                kwargs["response_format"] = response_format

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            logger.debug(f"LLM response: {len(content)} chars, tokens={response.usage.total_tokens}")

            return content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate completion with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            response_format: Optional response format

        Returns:
            Generated text response
        """
        temperature = temperature if temperature is not None else self.temperature

        try:
            logger.debug(f"LLM call with history: {len(messages)} messages")

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = await self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content

            logger.debug(f"LLM response: {len(content)} chars")

            return content

        except Exception as e:
            logger.error(f"LLM generation with history failed: {e}")
            raise


# Global LLM client instance
llm_client = LLMClient()
