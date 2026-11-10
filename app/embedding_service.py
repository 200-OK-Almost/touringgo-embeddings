import os
from pathlib import Path
from threading import Lock

from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

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
        return self._generate(texts, "passage")

    def generate_queries(self, texts: list[str]) -> list[list[float]]:
        return self._generate(texts, "query")

    def _generate(
        self,
        texts: list[str],
        prefix: str
    ) -> list[list[float]]:

        prefixed_texts = [
            f"{prefix}: {text}"
            for text in texts
        ]

        with self.lock:
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )

        return embeddings.tolist()