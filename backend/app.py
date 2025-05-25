from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app) 

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

index = faiss.read_index(os.path.join(DATA_DIR, "location_embeddings.faiss"))
with open(os.path.join(DATA_DIR, "location_metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 15)

    vec = model.encode([prompt], normalize_embeddings=True)
    vec = np.array(vec).astype("float32")
    scores, indices = index.search(vec, top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)