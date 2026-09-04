from fastapi import APIRouter

from app.schemas.query import QueryRequest, QueryResponse
from app.services.retrieval import retrieve_documents
from app.services.generation import generate_answer
from app.core.config import settings


router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    retrieved_chunks = retrieve_documents(
        request.question,
        top_k=settings.top_k
    )

    answer = generate_answer(
        request.question,
        retrieved_chunks
    )

    sources = list(dict.fromkeys(
    f"Page {chunk['page_number']}"
    for chunk in retrieved_chunks
    ))

    return QueryResponse(
        answer=answer,
        sources=sources
    )