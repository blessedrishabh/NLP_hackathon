import os
import json
from sentence_transformers import SentenceTransformer
import chromadb

def load_sections(parsed_file):
    with open(parsed_file, 'r', encoding='utf-8') as f:
        sections = json.load(f)
    return sections

def main(parsed_file, db_dir="chroma_db"):
    sections = load_sections(parsed_file)

    texts = []
    metadatas = []
    ids = []

    for i, section in enumerate(sections):
        text = f"{section['title']}\n{section['content']}"
        texts.append(text)
        metadatas.append({
            "section": str(section.get("section") or ""),   # ← was "section_id", but your JSON uses "section"
            "title":   str(section.get("title") or ""),
            "parent":  str(section.get("parent") or "")
        })
        ids.append(str(i))

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # ✅ Use PersistentClient — replaces the deprecated Settings approach
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection("sections")

    embeddings = model.encode(texts, show_progress_bar=True).tolist()  # ✅ Convert to list

    # Batch upsert (safer than add if re-running)
    collection.upsert(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ {len(texts)} embeddings stored in ChromaDB at './{db_dir}'")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embed_and_store.py parsed_sections.json")
    else:
        main(sys.argv[1])