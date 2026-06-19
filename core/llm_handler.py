"""
ResearchMind AI – LLM Handler
Unified interface for three free LLM providers:

  • Groq   → llama-3.1-70b-versatile   (free: console.groq.com, 30 req/min)
  • Gemini → gemini-1.5-flash          (free: ai.google.dev, 15 req/min)
  • Ollama → llama3.2 / any local model (offline: ollama.ai)
"""

import json
from typing import Generator, Optional, List, Dict

from core.config import (
    LLM_PROVIDER,
    GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
)


class LLMHandler:
    """
    Thin wrapper around provider SDKs.
    Usage:
        llm = LLMHandler()
        # non-streaming
        text = llm.generate("summarise this paper", system_prompt="You are ...")
        # streaming
        for token in llm.stream("explain the method"):
            print(token, end="", flush=True)
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or LLM_PROVIDER).lower()
        self._client  = None
        self._setup()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self):
        if self.provider == "groq":
            if not GROQ_API_KEY:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. "
                    "Get a free key at https://console.groq.com"
                )
            from groq import Groq
            self._client = Groq(api_key=GROQ_API_KEY)

        elif self.provider == "gemini":
            if not GEMINI_API_KEY:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. "
                    "Get a free key at https://ai.google.dev"
                )
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._client = genai.GenerativeModel(GEMINI_MODEL)

        elif self.provider == "ollama":
            # No SDK needed – plain HTTP
            pass

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                "Choose: groq | gemini | ollama"
            )

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """Return full response as a string (blocking) with fallback to Ollama."""
        try:
            if self.provider == "groq":
                return self._groq_generate(prompt, system_prompt, max_tokens, temperature)
            if self.provider == "gemini":
                return self._gemini_generate(prompt, system_prompt, max_tokens, temperature)
            if self.provider == "ollama":
                return self._ollama_generate(prompt, system_prompt, max_tokens, temperature)
        except Exception as e:
            if self.provider != "ollama":
                print(f"Primary LLM ({self.provider}) failed: {e}. Falling back to Ollama.")
                return self._ollama_generate(prompt, system_prompt, max_tokens, temperature)
            raise e

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """Yield tokens as they are produced with fallback to Ollama."""
        try:
            if self.provider == "groq":
                yield from self._groq_stream(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "gemini":
                yield from self._gemini_stream(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "ollama":
                yield from self._ollama_stream(prompt, system_prompt, max_tokens, temperature)
        except Exception as e:
            if self.provider != "ollama":
                print(f"Primary LLM ({self.provider}) failed during stream: {e}. Falling back to Ollama.")
                yield from self._ollama_stream(prompt, system_prompt, max_tokens, temperature)
            else:
                raise e

    def provider_info(self) -> Dict:
        model_map = {
            "groq":   GROQ_MODEL,
            "gemini": GEMINI_MODEL,
            "ollama": OLLAMA_MODEL,
        }
        return {"provider": self.provider, "model": model_map.get(self.provider, "unknown")}

    # ── Groq ──────────────────────────────────────────────────────────────────

    def _groq_messages(self, prompt, system_prompt):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _groq_generate(self, prompt, system_prompt, max_tokens, temperature):
        resp = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=self._groq_messages(prompt, system_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def _groq_stream(self, prompt, system_prompt, max_tokens, temperature):
        stream = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=self._groq_messages(prompt, system_prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _gemini_prompt(self, prompt, system_prompt):
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    def _gemini_generate(self, prompt, system_prompt, max_tokens, temperature):
        import google.generativeai as genai
        gen_cfg = genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature
        )
        resp = self._client.generate_content(
            self._gemini_prompt(prompt, system_prompt),
            generation_config=gen_cfg,
        )
        return resp.text or ""

    def _gemini_stream(self, prompt, system_prompt, max_tokens, temperature):
        import google.generativeai as genai
        gen_cfg = genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature
        )
        for chunk in self._client.generate_content(
            self._gemini_prompt(prompt, system_prompt),
            generation_config=gen_cfg,
            stream=True,
        ):
            if chunk.text:
                yield chunk.text

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _ollama_payload(self, prompt, system_prompt, max_tokens, temperature, stream):
        return {
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "system":  system_prompt,
            "stream":  stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    def _ollama_generate(self, prompt, system_prompt, max_tokens, temperature):
        import requests
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=self._ollama_payload(prompt, system_prompt, max_tokens, temperature, False),
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _ollama_stream(self, prompt, system_prompt, max_tokens, temperature):
        import requests
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=self._ollama_payload(prompt, system_prompt, max_tokens, temperature, True),
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("response"):
                        yield data["response"]
