import os
import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize API keys and configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "aurora-messages")

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it does not exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
    )

index = pc.Index(PINECONE_INDEX)

# Aurora messages API endpoint
AURORA_API = "https://november7-730026606190.europe-west1.run.app/messages/"

# Initialize FastAPI app
app = FastAPI(title="Aurora Question Answering API", version="1.0")

# Response model
class Answer(BaseModel):
    answer: str


# Fetch and index data from Aurora API
def build_index():
    try:
        response = requests.get(AURORA_API)
        data = response.json()
        messages = data.get("items", [])
        if not messages:
            print("No messages found from Aurora API.")
            return

        vectors = []
        for m in messages:
            text = f"{m['user_name'].strip().lower()}: {m['message'].strip().lower()}"
            emb = client.embeddings.create(model="text-embedding-3-small", input=text)
            vectors.append((
                m["id"],
                emb.data[0].embedding,
                {"user_name": m["user_name"], "message": m["message"]}
            ))

        index.upsert(vectors)
        print(f"Indexed {len(vectors)} messages successfully.")

    except Exception as e:
        print(f"Error while indexing data: {e}")


# Build index automatically on startup
@app.on_event("startup")
def startup_event():
    build_index()


# Root endpoint for health check
@app.get("/")
def root():
    return {"message": "Aurora Q&A API is running. Visit /docs to test the /ask endpoint."}


# Endpoint to manually reindex data
@app.post("/reindex")
def reindex():
    build_index()
    return {"message": "Reindexed data successfully."}


# Question answering endpoint
@app.get("/ask", response_model=Answer)
def ask(question: str = Query(..., description="Ask a question about member data")):
    try:
        # Create embedding for the user question
        q_emb = client.embeddings.create(model="text-embedding-3-small", input=question.lower())
        q_vector = q_emb.data[0].embedding

        # Query Pinecone for similar messages
        results = index.query(vector=q_vector, top_k=10, include_metadata=True)
        matches = results.get("matches", [])
        if not matches:
            return {"answer": "No relevant information found in the data."}

        # Filter top results by similarity score
        strong_matches = [m for m in matches if m["score"] > 0.65]
        if not strong_matches:
            strong_matches = matches

        # Prepare context from top matching messages
        context = "\n".join([
            f"{m['metadata']['user_name']}: {m['metadata']['message']}"
            for m in strong_matches
        ])

        print("\nContext retrieved:")
        print(context)
        print("--------------------------------------------------")

        # Construct reasoning prompt for GPT
        prompt = f"""
        You are Aurora, an intelligent assistant that answers user questions using member data.

        Use the provided context to answer the question factually and clearly.

        If the question refers to something not mentioned (for example, a trip to London),
        identify what related information is present (for example, a trip to Santorini)
        and explain that instead. Avoid speculation and only use information from the context.

        Context:
        {context}

        Question: {question}

        Provide a concise, factual answer.
        """

        # Generate completion from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )

        answer = completion.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        print(f"Error while processing question: {e}")
        return {"answer": "An error occurred while processing your question."}
