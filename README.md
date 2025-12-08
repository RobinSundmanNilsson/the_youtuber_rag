# The YouTuber RAG

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions based on a curated set of YouTube video transcripts. The goal is to create an assistant that responds in the style of the YouTuber while grounding its answers strictly in the provided transcript dataset.

---

## 🚀 Project Overview

The system consists of three main components:

### **1. Backend (FastAPI + PydanticAI + LanceDB)**
- Ingestion pipeline that loads transcript `.md` files and embeds them using **Gemini embeddings** via LanceDB’s registry (`gemini-embedding-001`).
- Vectors and metadata are stored in **LanceDB** (`knowledge_base/transcripts.lance`).
- A **PydanticAI Agent** (Gemini 2.5 Flash) retrieves relevant transcripts and generates grounded answers.
- The FastAPI endpoint `/rag/query` exposes the RAG pipeline for external use.

### **2. Frontend (Streamlit)**
- A clean and simple dashboard where users can:
  - Ask questions
  - View generated answers
  - See which transcripts were used as sources

### **3. Optional Deployment**
- Includes an **Azure Function App** template (`function_app.py`) suitable for deploying the API.
- Local testing using FastAPI and Streamlit works out-of-the-box.

---

## 📁 Project Structure

```
backend/
    constants.py         # Paths and global settings
    data_models.py       # Pydantic models and LanceDB schema (Prompt, RagResponse, Transcript)
    rag.py               # RAG pipeline (retrieval + PydanticAI agent)
frontend/
    app.py               # Streamlit dashboard
api.py                   # FastAPI server exposing /rag/query
ingestion.py             # Loads markdown transcripts → LanceDB
function_app.py          # Azure Function App (optional)
knowledge_base/          # LanceDB vector store
data/                    # Source .md transcripts
```

---

## 🔧 Installation & Setup

### 1. Create and activate virtual environment

```bash
uv init
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv sync
```

Dependencies include:
- FastAPI, Uvicorn
- Streamlit
- LanceDB
- PydanticAI
- google-generativeai (Gemini client)
- Requests
- azure-functions (for deployment)

### 3. Environment variables

Create `.env` with:

```
GOOGLE_API_KEY=your-google-api-key
AZURE_FUNCTION_APP_KEY=your-function-key (Optional)
```

---

## ⚡ Quickstart (local)

1) Create/activate venv and install deps (`uv init && source .venv/bin/activate && uv sync`).
2) Add `.env` with `GOOGLE_API_KEY=...` (and optionally `AZURE_FUNCTION_APP_KEY` if you’ll call a deployed Function).
3) Ingest transcripts: `uv run ingestion.py`.
4) Start local Function host (FastAPI via Azure Functions): `uv run func start`.
5) Run Streamlit UI: `uv run streamlit run frontend/app.py` (use the local `API_URL` in `frontend/app.py` when testing locally).

---

## 📥 Ingest Data Into LanceDB

Before running the RAG system the first time, run:

```bash
uv run ingestion.py
```

This script:
- Reads all `.md` files in `data/`
- Creates Gemini embeddings via LanceDB
- Stores them in `knowledge_base/transcripts.lance`

---

## 🧠 Running the Backend API

Start FastAPI:

```bash
uv run uvicorn api:app --reload
```

Open docs:

```
http://127.0.0.1:8000/docs
```

Use the `/rag/query` endpoint:

Example payload:

```json
{"prompt": "Explain what FastAPI CRUD is"}
```

---

## 🎨 Running the Streamlit Frontend

```bash
uv run streamlit run frontend/app.py
```

Features:
- Text input for user questions
- Pretty display of answers
- Dynamic list of transcript sources used for grounding

API URL configuration (`frontend/app.py`):
- Local Functions: uncomment `API_URL = "http://127.0.0.1:7071/rag/query"` to hit your locally running `uv run func start`.
- Deployed Azure Function: keep the env-based config. Set `AZURE_FUNCTION_APP_KEY` in your `.env` (value from Azure Function App → Functions → App Keys) and the app will call `https://<YOUR-FUNCTION-APP-NAME>.azurewebsites.net/rag/query?code=${AZURE_FUNCTION_APP_KEY}`.

---

## 🔍 How the RAG System Works

### **Embedding**
User query → embedded with Gemini (`gemini-embedding-001`) via LanceDB registry → vector search in LanceDB.

### **Retrieval**
Top-k relevant transcript chunks returned via a PydanticAI tool (`retrieve_top_transcripts`).

### **Generation**
Gemini model receives:
- System prompt aligning tone with the YouTuber
- Retrieved transcript blocks
- User query

Model outputs `{ answer, sources }` as a structured Pydantic object.

---

## 🛠 API Response Format

Example:

```json
{
  "answer": "To build a FastAPI CRUD app...",
  "sources": [
    {
      "video_id": "Fastapi CRUD app",
      "title": "Fastapi CRUD app",
      "score": null
    }
  ]
}
```

---

## 🧪 Testing Locally

You can test with:

```bash
curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is PydanticAI?"}'
```

---

## ☁️ Azure Deployment (Optional)

`function_app.py` contains a blueprint for deploying the FastAPI RAG server as an **Azure Function App**.

`local.settings.json` maintains non-production configuration.

Local Functions runtime (smoke-test before deploy):

```bash
uv run func start
```

Make sure `GOOGLE_API_KEY` is available locally (e.g. via `.env`) so the local Functions host can call Gemini.

Deploy flow:
- Deploy the Function App (same FastAPI app proxied via `function_app.py`).
- In Azure Portal: Function App → Settings → Environment Variables → add `GOOGLE_API_KEY` (no quotes) and restart.
- To call the deployed API from local Streamlit, grab an app key under Function App → Functions → App Keys, set it in your local `.env` as `AZURE_FUNCTION_APP_KEY=...`, and the app will automatically append it as `?code=${AZURE_FUNCTION_APP_KEY}` to the API URL.

---

## ⭐ Future Improvements

- Add chunking to make retrieval more granular.
- Store additional metadata (timestamps, categories).
- Support chat history (conversational RAG).
- Add vector store introspection dashboard.

---

## 🤝 Credits

This project was built as part of a university assignment to explore practical RAG systems, vector databases, and LLM orchestration using **PydanticAI** and **Gemini**.

---
