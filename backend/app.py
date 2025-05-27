from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import io
import os 


app = Flask(__name__)
CORS(app)


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

FAISS_URL = os.environ["FAISS_URL"]
PKL_URL   = os.environ["PKL_URL"]


faiss_resp = requests.get(FAISS_URL)
faiss_resp.raise_for_status()
faiss_bytes = faiss_resp.content
faiss_arr = np.frombuffer(faiss_bytes, dtype='uint8')
index = faiss.deserialize_index(faiss_arr)

pkl_resp = requests.get(PKL_URL)
pkl_resp.raise_for_status()
metadata = pickle.loads(pkl_resp.content)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    prompt = data.get("prompt", "")
    top_k = data.get("top_k", 10)

    vec = model.encode([prompt], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(vec, top_k)

    results = [metadata[i] for i in indices[0]]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)