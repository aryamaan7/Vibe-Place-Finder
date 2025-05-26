import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

def main():
    input_path = "data/us_location_vibes.csv"
    df = pd.read_csv(input_path)
    df["label"] = df["Location"] + ", " + df["state"]
    model = SentenceTransformer(MODEL_NAME)

    print("🔄 Embedding descriptions...")
    embeddings = embed_texts(df["description"].tolist(), model)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1]) 
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/location_embeddings.faiss")

    metadata = df[["label", "description", "article_url"]].to_dict(orient="records")
    with open("data/_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved {len(df)} vectors to FAISS and metadata to .pkl")

if __name__ == "__main__":
    main()