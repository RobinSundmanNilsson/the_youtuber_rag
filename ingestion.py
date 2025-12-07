import os
from pathlib import Path

import lancedb
from dotenv import load_dotenv
from google import generativeai as genai

from backend.constants import VECTOR_DATABASE_PATH, DATA_PATH
from backend.data_models import Transcript, EMBEDDING_DIM

TABLE_NAME = "transcripts"

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

genai.configure(api_key=GOOGLE_API_KEY)
EMBEDDING_MODEL = "text-embedding-004"

def setup_vector_db(path):
    Path(path).mkdir(exist_ok=True)
    vector_db = lancedb.connect(uri=path)
    if TABLE_NAME in vector_db.table_names():
        vector_db.drop_table(TABLE_NAME)
    vector_db.create_table(TABLE_NAME, schema=Transcript, exist_ok=True)
    return vector_db

def embed_text(text: str) -> list[float]:
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    embedding = response.get("embedding")
    if not embedding:
        raise RuntimeError("Embedding service returned an empty embedding for document.")
    if len(embedding) != EMBEDDING_DIM:
        raise RuntimeError(f"Embedding length mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}")
    return embedding

def ingest_docs_to_vector_db(table):
    rows = []

    for file in DATA_PATH.glob("*.md"):
        text = file.read_text(encoding="utf-8")

        video_id = file.stem
        title = file.stem.replace("_", " ")

        emb = embed_text(text)

        rows.append(
            {
                "video_id": video_id,
                "title": title,
                "text": text,
                "embedding": emb,
            }
        )

    table.delete("true")
    table.add(rows)

    print("Ingested", len(rows), "transcripts")
    print(table.to_pandas().head())


if __name__ == "__main__":
    vector_db = setup_vector_db(VECTOR_DATABASE_PATH)
    ingest_docs_to_vector_db(vector_db[TABLE_NAME])
