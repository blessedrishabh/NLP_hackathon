"""
embed_and_store.py  (v3)
========================
+ Added section classification (label) for smarter retrieval
"""

import os
import re
import sys
import json
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def table_to_text(table: dict) -> str:
    lines = []
    table_id = table.get("table_id", "Table")
    lines.append(table_id)

    rows = table.get("data", [])
    for row in rows:
        if not any(str(c).strip() for c in row):
            continue
        lines.append(" | ".join(str(c).strip() for c in row))

    return "\n".join(lines)


def make_section_text(section: dict) -> str:
    parts = [section["title"]]
    if section.get("content"):
        parts.append(section["content"])
    return "\n".join(parts)


def make_safe_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)


# ---------------------------------------------------------------------------
# NEW: Section classifier
# ---------------------------------------------------------------------------

def classify_section(title: str, content: str) -> str:
    text = (title + " " + content).lower()

    if any(x in text for x in ["mix", "batch", "curing", "compaction", "placing"]):
        return "procedure"
    if any(x in text for x in ["equipment", "mixer", "vibrator", "pump"]):
        return "equipment"
    if any(x in text for x in ["test", "strength", "quality", "cube"]):
        return "quality"
    if any(x in text for x in ["is code", "bis", "standard", "specification"]):
        return "references"
    if any(x in text for x in ["safety", "ppe", "hazard"]):
        return "health_safety"

    return "other"


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

def build_chunks(sections: list[dict]) -> list[dict]:
    chunks = []
    counter = 0

    def next_id(label: str) -> str:
        nonlocal counter
        uid = f"{counter:04d}_{make_safe_id(label)}"
        counter += 1
        return uid

    for section in sections:
        sec_id  = section.get("section", "")
        title   = section.get("title", "")
        content = section.get("content", "")
        parent  = str(section.get("parent") or "")
        tables  = section.get("tables") or []
        has_tbl = len(tables) > 0

        # 🔹 classify once per section
        label = classify_section(title, content)

        # ---- 1. Section chunk --------------------------------------------
        sec_text = make_section_text(section)
        if sec_text.strip():
            chunks.append({
                "id": next_id(f"sec_{sec_id}"),
                "text": sec_text,
                "metadata": {
                    "chunk_type": "section",
                    "section": sec_id,
                    "title": title,
                    "parent": parent,
                    "has_tables": str(has_tbl),
                    "label": label,   # ✅ added
                },
            })

        # ---- 2. Table chunks ---------------------------------------------
        for idx, table in enumerate(tables):
            tbl_text = table_to_text(table)
            table_id = table.get("table_id", f"table_{idx}")

            contextualised = (
                f"Section {sec_id} – {title}\n"
                f"{tbl_text}"
            )

            chunks.append({
                "id": next_id(f"tbl_{sec_id}_{idx}_{table_id}"),
                "text": contextualised,
                "metadata": {
                    "chunk_type": "table",
                    "section": sec_id,
                    "title": title,
                    "parent": parent,
                    "table_id": table_id,
                    "has_tables": "true",
                    "label": label,   # ✅ added
                },
            })

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(parsed_file: str, db_dir: str = "chroma_db"):
    with open(parsed_file, "r", encoding="utf-8") as f:
        sections = json.load(f)

    chunks = build_chunks(sections)

    print(f"Built {len(chunks)} chunks  "
          f"({sum(1 for c in chunks if c['metadata']['chunk_type']=='section')} section, "
          f"{sum(1 for c in chunks if c['metadata']['chunk_type']=='table')} table)")

    # ---- Embed ------------------------------------------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # ---- Store ------------------------------------------------------------
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection("sections")

    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"✅ {len(chunks)} chunks stored in ChromaDB at './{db_dir}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embed_and_store.py parsed_sections.json [chroma_db_dir]")
        sys.exit(1)

    parsed_file = sys.argv[1]
    db_dir = sys.argv[2] if len(sys.argv) > 2 else "chroma_db"
    main(parsed_file, db_dir)