import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index_and_metadata():
    index = faiss.read_index("data/location_embeddings.faiss")
    with open("data/location_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_query(text, model):
    emb = model.encode([text], normalize_embeddings=True)
    return np.array(emb).astype("float32")

def search(query, top_k=15):
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    query_vec = embed_query(query, model)
    scores, indices = index.search(query_vec, top_k)

    print(f"\n🔍 Top {top_k} matches for: \"{query}\"")
    print("-" * 60)

    for i, idx in enumerate(indices[0]):
        match = metadata[idx]
        print(f"{i+1}. {match['label']}")
        print(f"   {match['description']}")
        print(f"   🌐 {match['article_url']}")
        print()

if __name__ == "__main__":
    user_query = input("Enter a vibe you're looking for: ")
    search(user_query, top_k=15)