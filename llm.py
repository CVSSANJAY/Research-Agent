"""One wrapper for Groq, Gemini, OpenRouter and Ollama, with retries and model fallback."""
import os
import time
from openai import OpenAI, APIStatusError, APIConnectionError

# provider -> (base_url, api key env var, list of models)
# EDIT THE MODEL LISTS HERE. They are tried in order: if the first model is retired (404)
# or its daily quota runs out, the next one is used automatically.
PROVIDERS = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        [
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-flash-lite-latest",
            "gemini-2.5-flash-lite",
        ],
    ),
    "groq": (
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        ["meta-llama/llama-3.3-70b-instruct:free"],
    ),
    "ollama": ("http://localhost:11434/v1", None, ["qwen2.5:7b"]),
}


class LLM:
    def __init__(self, provider: str, api_key: str = None):
        base_url, key_env, default_models = PROVIDERS[provider]
        api_key = api_key or (os.getenv(key_env, "") if key_env else "ollama")
        if key_env and not api_key:
            raise RuntimeError(f"Set {key_env} in your .env file")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # The list above is used. A MODEL line in .env is optional and overrides it.
        raw = os.getenv("MODEL", "")
        names = [m.strip() for m in raw.split(",") if m.strip()] or default_models
        self.models = [m.removeprefix("models/") for m in names]
        self.idx = 0
        self.min_interval = float(os.getenv("CALL_DELAY", "0"))
        self._last = 0.0

    @property
    def model(self):
        return self.models[self.idx]

    def _next_model(self) -> bool:
        """Move to the next model in the list. Returns False if there is none left."""
        if self.idx + 1 < len(self.models):
            self.idx += 1
            print(f"  (switching to model: {self.model})")
            return True
        return False

    def chat(self, messages, tools=None, temperature=0.2, json_mode=False):
        """Returns the assistant message. Retries on rate limits; falls back to the next model
        when a model is gone (404) or its daily quota is used up."""
        attempt = 0
        while True:
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                kwargs = dict(model=self.model, messages=messages, temperature=temperature)
                if tools:
                    kwargs["tools"] = tools
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                self._last = time.time()
                return resp.choices[0].message
            except APIStatusError as e:
                self._last = time.time()
                text = str(e).lower()
                if e.status_code == 400 and json_mode:      # provider rejects JSON mode
                    json_mode = False
                    continue
                daily = "per day" in text or "tpd" in text or "daily" in text
                if (daily or e.status_code == 404) and self._next_model():
                    continue
                if daily:
                    raise RuntimeError(
                        "Daily quota used up for every model in MODEL. Add another model name, "
                        "switch PROVIDER in .env, or wait for the reset.") from e
                if e.status_code in (413, 429, 500, 502, 503) and attempt < 4:
                    attempt += 1
                    delay = 15 * attempt
                    print(f"  (API busy or rate limited, retrying in {delay}s)")
                    time.sleep(delay)
                    continue
                raise
            except APIConnectionError:
                if attempt < 4:
                    attempt += 1
                    time.sleep(5)
                    continue
                raise