from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import EmbeddingRequest, EmbeddingResponse
from app.embedding_service import EmbeddingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialización: se carga el modelo al arrancar.
    app.state.embedding_service = EmbeddingService()

    yield

    # Finalización.
    del app.state.embedding_service


app = FastAPI(
    title="TouringGO Embeddings API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "embedding_service")
    }


@app.post(
    "/embeddings/documents",
    response_model=EmbeddingResponse
)
def generate_documents(request: EmbeddingRequest):

    service = app.state.embedding_service

    try:
        embeddings = service.generate_documents(request.texts)

        return EmbeddingResponse(
            embeddings=embeddings,
            dimensions=service.dimensions,
            count=len(embeddings)
        )

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail="Error generating document embeddings"
        ) from ex


@app.post(
    "/embeddings/queries",
    response_model=EmbeddingResponse
)
def generate_queries(request: EmbeddingRequest):

    service = app.state.embedding_service

    try:
        embeddings = service.generate_queries(request.texts)

        return EmbeddingResponse(
            embeddings=embeddings,
            dimensions=service.dimensions,
            count=len(embeddings)
        )

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail="Error generating query embeddings"
        ) from ex