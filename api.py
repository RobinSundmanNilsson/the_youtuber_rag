from fastapi import FastAPI
from backend.rag import rag_agent
from backend.data_models import Prompt, RagResponse

app = FastAPI()


@app.post("/rag/query", response_model=RagResponse)
async def query_documentation(query: Prompt) -> RagResponse:
    result = await rag_agent.run(query.prompt)

    return result.output