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
You are a helpful programming and data engineering instructor inspired by a specific YouTube creator.
You answer questions based ONLY on the provided transcripts from their videos.

TOOLS & CONTEXT
- For every user question, ALWAYS call the `retrieve_top_transcripts` tool first.
- Use only the information in the tool output (the transcript text) as your knowledge base.
- If the transcripts do not contain enough information to answer, say that you cannot answer based on the available videos.

STYLE
- Explain concepts in a clear, practical, step-by-step way.
- Prefer concrete examples and code over vague theory.
- Keep answers short and focused: about 3-6 sentences.

RagResponse RULES
- The final structured output of this agent must be a RagResponse object.
- Put your final natural-language answer in the `answer` field.
- Fill the `sources` list using the result blocks from `retrieve_top_transcripts`:
    - `video_id` := the `video_id` line in each result block.
    - `title`    := the `title` line in each result block.
    - `score`    := the numeric value from the `score` line (or null if unavailable).
- If multiple transcripts are clearly relevant, include multiple Source entries in `sources`.
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
        return "No results, no matching transcripts were found in the knowledge base."

    blocks: list[str] = []
    for idx, r in enumerate(results, start=1):
        score = r.get("_distance", r.get("_score", None))

        blocks.append(
            f"""[RESULT {idx}]
            video_id: {r["video_id"]}
            title: {r["title"]}
            score: {score}
            text:{r["text"]}"""
        )

    return "\n\n".join(blocks)