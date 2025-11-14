import os
import requests
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "aurora-messages")

# Create Pinecone index if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENVIRONMENT")
        )
    )

index = pc.Index(index_name)

# External dataset API
AURORA_API = "https://november7-730026606190.europe-west1.run.app/messages/"

# FastAPI app
app = FastAPI(title="Aurora QA System")


# Response schema
class Answer(BaseModel):
    answer: str


# Root redirect → /docs
@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")


# Build Pinecone index on startup
@app.on_event("startup")
def build_index():
    response = requests.get(AURORA_API)
    data = response.json()
    messages = data.get("items", [])

    if not messages:
        return

    vectors = []
    for m in messages:
        text = f"{m['user_name']}: {m['message']}"
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        vectors.append((
            m["id"],
            emb.data[0].embedding,
            {"user_name": m["user_name"], "message": m["message"]}
        ))

    index.upsert(vectors)


# /ask endpoint
@app.get("/ask", response_model=Answer)
def ask(question: str = Query(..., description="Ask any question about members")):
    # Embed question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=q_emb,
        top_k=5,
        include_metadata=True
    )

    matches = results.get("matches", [])

    # If no retrieved contexts → fallback
    if len(matches) == 0:
        return {"answer": "I cannot find any relevant information in the available messages."}

    # Build context from all matches
    context = "\n".join(
        f"- {m['metadata']['user_name']}: {m['metadata']['message']}"
        for m in matches
    )

    # LLM prompt
    prompt = f"""
You are Aurora's intelligent concierge assistant.

You will answer the question using ONLY the member messages provided below.

Guidelines:
1. If the context includes related information, use it to provide the best possible specific answer.
2. If the question asks something not explicitly available, but related info exists, acknowledge what IS known.
3. Only if there is absolutely no related information, respond:
   "Make Reasoning Corrctly based on context retrieved if it has partial relevance or no relevance."

Context:
{context}

Question: {question}

Answer:
"""

    # Generate answer
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = completion.choices[0].message.content.strip()
    return {"answer": answer}
