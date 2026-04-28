"""
embed_and_store.py  (v2)
========================
Improvements over v1:
- Tables are embedded as SEPARATE chunks (not silently dropped)
- Table rows are serialised to readable pipe-delimited text so the embedding
  model can understand column relationships (e.g. "IS Sieve 10mm | 25 to 55")
- Section text chunks no longer contain raw table noise
- Metadata is richer: chunk_type, has_tables, table_id
- Duplicate-safe: uses upsert with stable deterministic IDs
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
    """
    Convert a structured table dict into readable pipe-delimited text.

    Example output:
        Table 4.1 - Graded Stone Aggregate or Gravel
        IS Sieve Designation | 40 mm | 20 mm | 16 mm | 12.5 mm
        80 mm | 100 | - | - | -
        40 mm | 95 to 100 | 100 | - | -
        ...
    """
    lines = []
    table_id = table.get("table_id", "Table")
    lines.append(table_id)                          # caption as first line

    rows = table.get("data", [])
    for row in rows:
        # Skip rows that are entirely empty
        if not any(str(c).strip() for c in row):
            continue
        lines.append(" | ".join(str(c).strip() for c in row))

    return "\n".join(lines)


def make_section_text(section: dict) -> str:
    """Plain text for a section chunk (title + prose, NO table rows)."""
    parts = [section["title"]]
    if section.get("content"):
        parts.append(section["content"])
    return "\n".join(parts)


def make_safe_id(raw: str) -> str:
    """ChromaDB requires IDs with no special chars beyond hyphens/underscores."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw)


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

def build_chunks(sections: list[dict]) -> list[dict]:
    """
    Returns a flat list of chunks, each with:
        id          – stable string ID for ChromaDB upsert
        text        – what gets embedded
        metadata    – stored alongside the vector

    Two chunk types:
        "section"  – prose text of a section
        "table"    – one table belonging to a section

    IDs are prefixed with a zero-padded global counter so they are always
    unique, even when two section IDs normalise to the same safe string
    (e.g. "5.0" and "5-0" both become "5_0" after sanitisation).
    """
    chunks = []
    counter = 0   # monotonically increasing, guarantees no collisions

    def next_id(label: str) -> str:
        nonlocal counter
        uid = f"{counter:04d}_{make_safe_id(label)}"
        counter += 1
        return uid

    for section in sections:
        sec_id    = section.get("section", "")
        title     = section.get("title", "")
        parent    = str(section.get("parent") or "")
        tables    = section.get("tables") or []
        has_tbl   = len(tables) > 0

        # ---- 1. Section prose chunk ----------------------------------------
        sec_text = make_section_text(section)
        if sec_text.strip():
            chunks.append({
                "id"  : next_id(f"sec_{sec_id}"),
                "text": sec_text,
                "metadata": {
                    "chunk_type": "section",
                    "section"   : sec_id,
                    "title"     : title,
                    "parent"    : parent,
                    "has_tables": str(has_tbl),   # ChromaDB metadata must be str/int/float
                },
            })

        # ---- 2. One chunk per table -----------------------------------------
        for idx, table in enumerate(tables):
            tbl_text = table_to_text(table)
            table_id = table.get("table_id", f"table_{idx}")

            # Include section title as context prefix so the embedding
            # knows WHAT the table is about, not just raw numbers.
            contextualised = (
                f"Section {sec_id} – {title}\n"
                f"{tbl_text}"
            )

            chunks.append({
                "id"  : next_id(f"tbl_{sec_id}_{idx}_{table_id}"),
                "text": contextualised,
                "metadata": {
                    "chunk_type": "table",
                    "section"   : sec_id,
                    "title"     : title,
                    "parent"    : parent,
                    "table_id"  : table_id,
                    "has_tables": "true",
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
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # ---- Store ------------------------------------------------------------
    client     = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection("sections")

    collection.upsert(
        ids        = [c["id"]       for c in chunks],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [c["metadata"] for c in chunks],
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
    db_dir      = sys.argv[2] if len(sys.argv) > 2 else "chroma_db"
    main(parsed_file, db_dir)