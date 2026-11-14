import os
from pathlib import Path
from threading import Lock

from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")

MODEL_PATH = os.getenv("MODEL_PATH")

class EmbeddingService:

    def __init__(self):
        if MODEL_PATH:
            model_location = Path(MODEL_PATH)
        else:
            model_location = MODEL_NAME

        # El modelo se carga una sola vez.
        # Si no existe localmente, Hugging Face lo descarga.
        self.model = SentenceTransformer(str(model_location))

        self.dimensions = self.model.get_sentence_embedding_dimension()

        # Evita ejecuciones simultáneas sobre la misma instancia.
        self.lock = Lock()

    def generate_documents(self, texts: list[str]) -> list[list[float]]:
        return self._generate(texts)

    def generate_queries(self, texts: list[str]) -> list[list[float]]:
        return self._generate(texts)

    def _generate(self, texts: list[str]) -> list[list[float]]:

        with self.lock:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )

        return embeddings.tolist()