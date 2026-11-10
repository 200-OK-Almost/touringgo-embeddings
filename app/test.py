import requests
import numpy as np

BASE_URL = "http://localhost:8084"

documents = [
    "Museos de arte y exposiciones culturales en Buenos Aires.",
    "Parques y espacios verdes para disfrutar de la naturaleza.",
    "Monumentos y edificios históricos de la ciudad.",
    "Restaurantes de comida italiana.",
]

queries = [
    "Galerías de arte y muestras culturales.",
    "Lugares para caminar al aire libre y disfrutar de la naturaleza.",
    "Edificios históricos y monumentos emblemáticos.",
    "Dónde comer pastas y pizza.",
]

# Generate document embeddings
doc_response = requests.post(
    f"{BASE_URL}/embeddings/documents",
    json={"texts": documents}
)
doc_response.raise_for_status()
doc_vectors = np.array(doc_response.json()["embeddings"])

# Generate query embeddings
query_response = requests.post(
    f"{BASE_URL}/embeddings/queries",
    json={"texts": queries}
)
query_response.raise_for_status()
query_vectors = np.array(query_response.json()["embeddings"])

# Both sets are normalized, so dot product = cosine similarity
similarities = query_vectors @ doc_vectors.T

for i, query in enumerate(queries):
    print(f"\nQuery: {query}")

    ranking = np.argsort(similarities[i])[::-1]

    for index in ranking:
        print(
            f"  {similarities[i][index]:.4f} - "
            f"{documents[index]}"
        )