"""
Provider-agnostic LLM client.
Architecture: defines a common interface - swap providers by changing config, not code.
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

def generate(task_prompt: str, user_text: str) -> dict:
    """
    Send a completion request to the configured LLM provider.
    Returns dict with 'content' and 'tokens' keys.
    """

    if PROVIDER == "anthropic":
        return _call_anthropic(task_prompt, user_text)
    else :
        raise ValueError(f"Unsupported provider: {PROVIDER}. Supported: anthropic")

def _call_anthropic(task_prompt: str, user_text: str) -> dict:
    from anthropic import Anthropic, AuthenticationError, RateLimitError, APIConnectionError

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not configured")

    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=task_prompt,
            messages=[
                {"role": "user", "content": user_text}
            ],
            extra_body={"temperature": 0.7}
        )
        return {
            "content": response.content[0].text,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
            "model": response.model
        }
    except AuthenticationError:
        raise PermissionError("Invalid API key. Check your .env file.")
    except RateLimitError:
        raise RuntimeError("Rate limit hit. Wait a moment and try again.")
    except APIConnectionError:
        raise ConnectionError("Cannot connect to Anthropic. Check your internet.")

def get_provider_info() -> dict:
    return {"provider": PROVIDER, "model": MODEL}
