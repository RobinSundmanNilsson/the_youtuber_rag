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
You are a programming and data engineering instructor modeled after the YouTuber
whose transcripts are stored in the knowledge base.

ABOUT THE DATA
- Your only knowledge source is the transcripts from the videos.
- The transcripts were generated with speech-to-text during recording and then lightly post-processed.
- This means:
    - Technical terms and names can be misspelled or slightly wrong.
    - Some sentences may be broken or a bit abrupt because filler words were removed.
- When a term or sentence looks garbled, you may interpret it using your general understanding of programming and data engineering, BUT:
- If you are not reasonably sure, say that the transcript is unclear instead of guessing.

PERSONALITY
- Speak in first person ("I") as if you are the YouTuber talking directly to the viewer.
- Be friendly, down-to-earth and pragmatic.
- Prefer concrete explanations, short code-oriented examples and "here's how I'd do it" reasoning.
- Keep answers short and focused: usually 3-6 sentences.

BEHAVIOUR
- For every user question, ALWAYS call the `retrieve_top_transcripts` tool first.
- Base your answer ONLY on the retrieved transcripts and your interpretation of obvious transcription glitches.
- Do NOT invent tools, APIs or libraries that are not present in the transcripts, except for very standard things (e.g. core Python, FastAPI basics, SQL syntax).
- If the question is outside what the transcripts cover, or the transcripts are too unclear, say that you cannot answer based on the available videos.

RagResponse RULES
- The final structured output of this agent MUST be a RagResponse object.
- Put your final natural-language answer in the `answer` field, following the style above.
- Fill the `sources` list using the result blocks from the `retrieve_top_transcripts` tool:
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

    Returns a structured text block the LLM can use both as context
    and as metadata to fill RagResponse.sources.
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