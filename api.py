from fastapi import FastAPI, HTTPException, status
from pydantic_ai.models.google import ModelHTTPError
from backend.rag import rag_agent
from backend.data_models import Prompt, RagResponse

app = FastAPI()


@app.post("/rag/query", response_model=RagResponse)
async def query_documentation(query: Prompt) -> RagResponse:
    try:
        result = await rag_agent.run(query.prompt)
    except ModelHTTPError as e:
        raise HTTPException(
            status_code=getattr(e, "status_code", status.HTTP_502_BAD_GATEWAY),
            detail={
                "error": "llm_request_failed",
                "model": getattr(e, "model_name", "unknown"),
                "status_code": getattr(e, "status_code", None),
                "message": getattr(e, "body", None) or str(e),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": str(e),
            },
        )

    return result.output
