from huggingface_hub import InferenceClient
from Src.App.config import settings
import logging

logger = logging.getLogger(__name__)

_client = None


def get_client():
    # Creates and returns the Hugging Face inference client.
    global _client

    if _client is None:
        # Stops if the Hugging Face token is missing.
        if not settings.hf_token:
            return None

        # Creates the client using the Hugging Face token.
        _client = InferenceClient(
            api_key=settings.hf_token,
        )

    return _client


def generate_text(prompt: str, max_new_tokens: int = 256) -> str:
    # Sends a prompt to the Hugging Face model and returns the response.
    client = get_client()

    # Returns an error message if the token is not configured.
    if client is None:
        return "[HF_TOKEN not configured]"

    try:
        # Sends the prompt to the selected Qwen model.
        completion = client.chat.completions.create(
            model=settings.hf_model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cybersecurity log analysis assistant. "
                        "Explain Windows event logs clearly and briefly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_new_tokens,
            temperature=0.1,
        )

        # Extracts the generated text from the model response.
        content = completion.choices[0].message.content

        if content is None:
            return "[No content generated]"

        return content.strip()

    except Exception as e:
        # Logs the error and returns a readable message.
        logger.exception("HF Inference API error")
        return f"[LLM API error: {repr(e)}]"