
import os
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()

# Initialize Groq client
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Gemini client
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Test 1: Gemini Embeddings
embedding_response = gemini_client.models.embed_content(
    model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
    contents="Hello Google Gemini",
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=int(os.getenv("EMBED_DIM", "768"))
    )
)

embedding = embedding_response.embeddings[0].values

print("Embedding dimension:", len(embedding))


# Test 2: Groq Chat
chat_response = groq_client.chat.completions.create(
    model=os.getenv("GROQ_CHAT_MODEL"),
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Explain RAG in one sentence."
        }
    ],
    max_tokens=200,
    temperature=0.1
)

print(
    "Groq Response:",
    chat_response.choices[0].message.content
)
