from huggingface_hub import InferenceClient
from Src.App.config import settings
import logging

logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None and settings.hf_token and settings.hf_model_id:
        _client = InferenceClient(
            model=settings.hf_model_id,
            token=settings.hf_token,
        )
        logger.info(f"Initialized HF Inference client for model: {settings.hf_model_id}")
    return _client


def generate_text(prompt: str, max_new_tokens: int = 256) -> str:
    if not settings.hf_token:
        logger.error("HF_TOKEN not set")
        return "[HF_TOKEN not configured]"
    
    if not settings.hf_model_id:
        logger.error("HF_MODEL_ID not set")
        return "[HF_MODEL_ID not configured]"

    client = get_client()
    if client is None:
        return "[Failed to initialize HF Inference client]"

    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.15,
        )

        if isinstance(response, str):
            return response.strip()

        return str(response)

    except Exception as e:
        logger.error(f"HF Inference API error: {e}")
        return f"[LLM API error: {str(e)}]"