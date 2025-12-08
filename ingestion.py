import lancedb
from backend.constants import VECTOR_DATABASE_PATH, DATA_PATH
from backend.data_models import Transcript
import time

TABLE_NAME = "transcripts"

def setup_vector_db(path):
    vector_db = lancedb.connect(uri = path)
    vector_db.create_table("transcripts", schema=Transcript, exist_ok=True)

    return vector_db

def ingest_docs_to_vector_db(table):
    ingested = 0

    for file in DATA_PATH.glob("*.md"):
        with open(file, "r") as f:
            text = f.read()

        video_id = file.stem
        table.delete(f"video_id = '{video_id}'")

        table.add([
            {
                "video_id": video_id,
                "title": file.stem,
                "text": text,
            }
        ])

        ingested += 1
        print("Ingested", ingested, "transcripts")
        print(table.to_pandas()[["video_id", "title"]])
        time.sleep(30)


if __name__ == "__main__":
    vector_db = setup_vector_db(VECTOR_DATABASE_PATH)
    ingest_docs_to_vector_db(vector_db[TABLE_NAME])
