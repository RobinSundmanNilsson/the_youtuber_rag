from backend.constants import VECTOR_DATABASE_PATH, DATA_PATH
import lancedb
from backend.data_models import Transcript
from pathlib import Path
from sentence_transformers import SentenceTransformer

TABLE_NAME = "transcripts"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def setup_vector_db(path):
    Path(path).mkdir(exist_ok=True)
    vector_db = lancedb.connect(uri=path)
    vector_db.create_table(TABLE_NAME, schema=Transcript, exist_ok=True)
    return vector_db

def ingest_docs_to_vector_db(table):
    rows = []

    for file in DATA_PATH.glob("*.md"):
        text = file.read_text(encoding="utf-8")

        video_id = file.stem
        title = file.stem.replace("_", " ")

        emb = embed_model.encode([text])[0]
        emb = emb.tolist()

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

# def ingest_docs_to_vector_db(table):
#     for file in DATA_PATH.glob("*.md"):
#         content = file.read_text(encoding="utf-8")

#         video_id = file.stem
#         title = file.stem.replace("_", " ")

#         table.delete(f"video_id = '{video_id}'")

#         table.add(
#             [
#                 {
#                     "video_id": video_id,
#                     "title": title,
#                     "text": content,
#                 }
#             ]
#         )

#     print(table.to_pandas().head())
#     print("Rows:", table.count_rows())


if __name__ == "__main__":
    vector_db = setup_vector_db(VECTOR_DATABASE_PATH)
    ingest_docs_to_vector_db(vector_db[TABLE_NAME])