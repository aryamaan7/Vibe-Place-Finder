import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("data/location_embeddings.faiss")
    with open("data/location_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_query(text, model):
    embedding = model.encode([text], normalize_embeddings=True)
    return np.array(embedding).astype("float32")

st.set_page_config(page_title="Place Vibes Finder", layout="centered")
st.title("🌍 Place Vibes Finder")
st.markdown("Describe your dream vibe, and we’ll match it to real U.S. towns and cities.")

query = st.text_input("What's your vibe?", placeholder="e.g., Quiet artsy desert town with mountain views")

if query:
    with st.spinner("Searching..."):
        model = load_model()
        index, metadata = load_faiss_index()
        query_vec = embed_query(query, model)

        scores, indices = index.search(query_vec, 15)
        st.subheader("Top Matches:")

        for i, idx in enumerate(indices[0]):
            match = metadata[idx]
            st.markdown(f"**{i+1}. {match['label']}**")
            st.markdown(match["description"])
            st.markdown(f"[🌐 Read more]({match['article_url']})")
            st.markdown("---")