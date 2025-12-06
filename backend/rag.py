from pathlib import Path

import lancedb
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer

from backend.constants import VECTOR_DATABASE_PATH
from backend.data_models import RagResponse

vector_db = lancedb.connect(str(VECTOR_DATABASE_PATH))
transcript_table = vector_db.open_table("transcripts")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(text: str) -> list[float]:
    """Embed a user query to the same vector space as transcripts."""
    return embed_model.encode([text])[0].tolist()

system_prompt = """
You are a helpful data engineering instructor based on the style of the Youtuber
whose transcripts you are given.

Always:
- First use the `retrieve_top_transcripts` tool to fetch relevant transcripts.
- Answer ONLY based on the retrieved transcripts. If the information is missing,
    say that you cannot answer from the available videos.
- Keep answers clear, direct and practical. Max around 6 sentences.
- When you produce the RagResponse:
- Put your actual answer in `answer`.
- Fill `sources` with the video_id and title of the transcripts you used
    (you can extract them from the tool output).
"""


rag_agent = Agent(
    model="google-gla:gemini-2.5-flash",
    retries=2,
    system_prompt=system_prompt,
    output_type=RagResponse,
)


@rag_agent.tool_plain
def retrieve_top_transcripts(query: str, k: int = 3) -> str:
    """
    Retrieve the top-k most relevant transcripts for a given query
    from the LanceDB 'transcripts' table.

    Returns a textual block the LLM can read and base its answer on.
    """
    query_embedding = embed_query(query)

    results = (
        transcript_table.search(query_embedding)
        .limit(k)
        .to_list()
    )

    if not results:
        return "No matching transcripts were found in the knowledge base."

    blocks: list[str] = []
    for r in results:
        blocks.append(
            f"Video ID: {r['video_id']}\n"
            f"Title: {r['title']}\n"
            f"Transcript:\n{r['text']}\n"
            "----"
        )

    return "\n\n".join(blocks)