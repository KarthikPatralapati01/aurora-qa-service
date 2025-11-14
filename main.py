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

# Create the index if it does not exist
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

AURORA_API = "https://november7-730026606190.europe-west1.run.app/messages/"

app = FastAPI(title="Aurora QA System")


class Answer(BaseModel):
    answer: str


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


# Index Aurora messages inside Pinecone at startup
@app.on_event("startup")
def build_index():
    response = requests.get(AURORA_API)
    data = response.json()
    messages = data.get("items", [])

    if not messages:
        print("No messages received from API.")
        return

    vectors = []
    for msg in messages:
        text = f"{msg['user_name']}: {msg['message']}"
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        vectors.append((
            msg["id"],
            embedding,
            {
                "user_name": msg["user_name"],
                "message": msg["message"]
            }
        ))

    index.upsert(vectors)
    print(f"Indexed {len(vectors)} messages.")


@app.get("/ask", response_model=Answer)
def ask(question: str = Query(..., description="Ask a question about Aurora members")):
    # Create embedding for the question
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=10,               # Better recall than 5
        include_metadata=True
    )

    matches = results.get("matches", [])

    # Build structured context
    context_blocks = []
    for m in matches:
        md = m["metadata"]
        context_blocks.append(
            f"- **{md['user_name']}**: {md['message']}"
        )

    context = "\n".join(context_blocks).strip()

  
        # If Pinecone returned zero matches, fallback
    if len(matches) == 0:
        return {
        "answer": "I cannot find any relevant information in the available messages."
    }


    # Improved prompt for reasoning
    prompt = f"""
You are Aurora's intelligent concierge assistant.

Your job is to answer ONLY using the context provided.

Guidelines:
1. If the question cannot be fully answered, but there is partial information, use it.
2. If the question asks something not explicitly stated, analyze the context and state what IS known.
   Example: “I don’t see any data saying Layla is traveling to London, but she DID plan a trip to Santorini.”
3. Only respond “I cannot find that information in the available messages.” if:
   - There is NO relevant mention of the person, OR
   - The context contains nothing related to the question.

Context:
{context}

Question: {question}

Provide a clear, concise answer:
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = completion.choices[0].message.content.strip()
    return {"answer": answer}
