# Aurora Natural-Language Q&A Service  
A lightweight question–answering API that allows users to ask natural-language questions about Aurora member messages.  
The service semantically indexes all messages from Aurora’s public API, stores embeddings in Pinecone, and uses OpenAI to generate accurate, context-aware answers.

##LOOM VIDEO Link: https://www.loom.com/share/30a1a99f6feb4a5dbe0cba4691714a53

## 📌 Overview  
This project implements a production-ready **semantic search + LLM reasoning** pipeline.  
Given any natural-language query such as:

- “When is Layla planning her trip to London?”  
- “How many cars does Vikram Desai have?”  
- “What are Amira’s favorite restaurants?”

The system retrieves relevant member messages from Pinecone, constructs contextual evidence, and generates a concise answer using OpenAI.

---

## 🚀 Live Demo  
Swagger Docs:  
```
https://<your-render-url>/docs
```

Ask Questions Through `/ask` Endpoint:  
```
GET https://<your-render-url>/ask?question=YOUR_QUESTION
```

---

## 📂 Project Structure
```
├── main.py               # Main FastAPI service
├── inspect_data.py       # Script to analyze dataset anomalies
├── requirements.txt      # Runtime dependencies
└── .gitignore            # Repo ignore list
```

---

## 🧠 Architecture

### High-Level Flow
```
                ┌────────────────────────────┐
                │  Aurora Public API (/messages)  
                └───────────────┬────────────┘
                                │ Fetch messages (startup)
                                ▼
                  ┌──────────────────────────┐
                  │ OpenAI Embedding Model   │
                  │ text-embedding-3-small   │
                  └──────────────┬───────────┘
                                 │ Generate semantic vectors
                                 ▼
                     ┌──────────────────────┐
                     │ Pinecone Vector DB   │
                     │ (aurora-messages)    │
                     └────────────┬─────────┘
                                  │ Store & index embeddings
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────────┐
       │ FastAPI `/ask` Endpoint                                    │
       │ 1. Embed question                                          │
       │ 2. Retrieve top-k matches from Pinecone                    │
       │ 3. Build context string                                    │
       │ 4. Generate answer using OpenAI (gpt-4o-mini)              │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
                             ┌───────────────────┐
                             │ JSON Answer Output│
                             │ { "answer": ... } │
                             └───────────────────┘
```

---

## ⚙️ How It Works

### 1. **Fetch Messages**
On application startup:

- The service calls the Aurora `/messages` API.
- Extracts all message text + user names.

### 2. **Embed Messages**
Each message is transformed into a semantic vector using:

```
text-embedding-3-small
```

### 3. **Store in Pinecone**
Vectors are upserted into the Pinecone index:

- dimension: **1536**
- metric: **cosine**
- type: **serverless**

### 4. **Answer Questions**
Users call:

```
GET /ask?question=...
```

Process:

1. Embed the question.
2. Query Pinecone for the top 5 similar messages.
3. Build contextual evidence.
4. Use GPT-4o-mini to generate the final answer.

### 5. **Inference Logic**
The model is instructed to:

- Use context responsibly  
- Give partial answers when possible  
- Avoid hallucination  
- Acknowledge missing details  
- Provide the best available reasoning  

Example:

**Q:** “How many cars does Vikram Desai have?”  
**A:** “Vikram mentioned a car, but the messages do not specify how many he owns.”

---

## 🧪 Example Queries

### Query:
```
GET /ask?question=When is Layla planning her trip to London?
```

### Possible Answer:
```
Layla mentioned planning a trip, but there is no information about a trip to London.
```


### Query:
```
GET /ask?question=How many cars does Vikram Desai have?
```

### Possible Answer:
```
Vikram mentioned a car, but the messages do not specify how many he owns.
```

---

## 🔍 Dataset Analysis (Anomalies & Insights)

A manual and scripted inspection of all 100 messages reveals:

### ✅ Observed Patterns
- Messages are short and action-oriented (booking, scheduling, requests).
- Multiple members have similar travel-related tasks.
- Most messages follow a consistent structure:  
  `"user_name": "...", "message": "...“`

### ⚠️ Detected Anomalies
1. **Inconsistent Name Formatting**  
   - Some names include apostrophes (e.g., *Lily O'Sullivan*).  
   - Others include Unicode characters (e.g., *Layla Kawaguchi*).  
   This requires consistent string handling.

2. **Messages Without Clear Intent**  
   - A few messages lack verbs or context (“Next Tuesday please.”)

3. **Duplicate Intent Across Users**  
   - Multiple users request similar tasks (hotel, flights, car pickup).  
   Useful for clustering but can confuse naive search.

4. **Ambiguity**  
   - Some messages imply actions (like owning a car) but do not provide explicit counts or details.

5. **No explicit dates or structured fields**  
   - All time expressions are free-text (“next Monday”, “first week of December”).

These insights help refine both semantic search and answer generation.

---

## 🛠️ Technologies Used
| Component | Purpose |
|----------|---------|
| **FastAPI** | API layer |
| **OpenAI GPT-4o-mini** | Answer generation |
| **OpenAI text-embedding-3-small** | Vector embeddings |
| **Pinecone Serverless** | Semantic search index |
| **Render** | Deployment |
| **Python 3.10+** | Runtime |

---

## 🚀 Deployment on Render

- Create a Web Service  
- Point to this repo  
- Use start command:  
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
- Add environment variables:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_ENVIRONMENT`
  - `PINECONE_INDEX`

---

## 📜 API Endpoints

### **GET /**
Redirects to Swagger UI.

### **GET /ask**
Ask a question.

**Query Parameter:**  
```
question: string (required)
```

**Response:**  
```json
{
  "answer": "..."
}
```

---

## 📎 Future Improvements

- Add re-ranking layer for higher accuracy  
- Support conversation history  
- Add structured extraction mode  
- Add message clustering for faster retrieval  
- Build UI chatbot interface  



---

## 📧 Contact  
**Venkata Karthik Patralapati**  
Email: venkatakarthik804@gmail.com  


