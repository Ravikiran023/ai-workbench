"""
Business logic layer - maps each task to its prompt templates.
Separated from routing (main.py) and LLM transport (llm_client.py).
"""

from .llm_client import generate

PROMPTS = {
    "summarize": "You are a concise summarizer. Summarize the user's text in 3-5 clear bullet points. Focus on the most important information.",
    "rewrite": "You are a professional editor. Rewrite the user's text in a clear, professional tone. Maintain the original meaning but improve clarity and readability.",
    "keypoints": "You are an analyst. Extract the key points from the user's text as a numbered list. Each point should be one clear sentence.",
    "explain": "You are a patient teacher. Explain the user's text in simple terms that a non-expert can understand. Use analogies where helpful.",
}

def process_task(task: str, text: str) -> dict:
    if task not in PROMPTS:
        raise ValueError(f"Unknown task: {task}, Available: {list(PROMPTS.keys())}")

    result = generate(PROMPTS[task], text)
    return {
        "task": task,
        "content": result["content"],
        "tokens_used": result["tokens"]
    }