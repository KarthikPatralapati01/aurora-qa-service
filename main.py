import os
import requests
from fastapi import FastAPI, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "aurora-messages")

# Create index if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENVIRONMENT"))
    )

index = pc.Index(index_name)

AURORA_API = "https://november7-730026606190.europe-west1.run.app/messages/"

app = FastAPI(title="Aurora QA System")

class Answer(BaseModel):
    answer: str


# Root → Redirect to /docs
@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")


# Build Pinecone index at startup
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


# Question-answering endpoint
@app.get("/ask", response_model=Answer)
def ask(question: str = Query(..., description="Ask any question about members")):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    q_vector = emb.data[0].embedding

    results = index.query(
        vector=q_vector,
        top_k=5,
        include_metadata=True
    )

    context = "\n".join(
        f"{r['metadata']['user_name']}: {r['metadata']['message']}"
        for r in results["matches"]
    )

    prompt = f"""


You are Aurora's intelligent concierge assistant.

Your job is to answer natural-language questions using ONLY the context provided below.

Rules:
1. If the context contains information about the member (even partial), use it to give the most accurate possible answer.
2. If the question asks for something specific that is NOT in the context, but there is related information, acknowledge what iS known.
3. Only when the context contains purely relevant information about the member or topic at all, respond:
   "I cannot find that information in the available messages."
Context:
{context}

Question: {question}
Answer:
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = completion.choices[0].message.content.strip()
    return {"answer": answer}
