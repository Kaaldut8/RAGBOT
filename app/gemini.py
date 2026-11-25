from functools import lru_cache
from google import genai
from google.genai import types
from app.config import settings
from app.observability import trace

@lru_cache
def client():
    return genai.Client(
        api_key=settings().genai_api_key
    )


@trace("embed")
def embed(text):
    s = settings()

    result = client().models.embed_content(
        models=s.gemini_embed_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=s.embed_dim
        )
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    return result.embeddings[0].values