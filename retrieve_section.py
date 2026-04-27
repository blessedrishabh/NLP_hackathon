import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

def retrieve(query, db_dir=None, top_k=3):
    if db_dir is None:
        db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    # Load model and ChromaDB
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection("sections")

    # Embed query
    query_emb = model.encode([query])
    # Search
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k
    )
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"Section: {meta['section']} | Title: {meta['title']}\n{doc}\n{'-'*40}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python retrieve_section.py 'Your query here'")
    else:
        retrieve(sys.argv[1])