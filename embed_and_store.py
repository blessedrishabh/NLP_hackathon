"""
embed_and_store.py  (v4 - Enhanced for New Pipeline)
=====================================================
Upgraded to work with parser v3:
+ Direct support for semantic chunks with metadata
+ Hierarchical context awareness
+ Improved classification using metadata
+ Hybrid embedding strategy (content + context)
+ Better table linking and context
+ Support for both v3 and v2 formats (backward compatible)
"""

import os
import re
import sys
import json
from typing import Optional, Dict, List
from sentence_transformers import SentenceTransformer
import chromadb


# ---------------------------------------------------------------------------
# Enhanced Classification with Metadata Awareness
# ---------------------------------------------------------------------------

SECTION_KEYWORDS = {
    "purpose": ["purpose", "scope", "objective", "aim", "goal", "intent"],
    "procedure": ["mix", "batch", "curing", "compaction", "placing", "pour", "casting",
                  "vibration", "mixing", "transport", "formwork", "reinforcement", "placing"],
    "equipment": ["equipment", "mixer", "vibrator", "pump", "machine", "plant", "tool",
                  "formwork", "scaffold", "props", "concrete mixer", "needle"],
    "quality": ["test", "strength", "quality", "cube", "sampling", "inspection", "acceptance",
                "compressive", "crushing", "ndt", "slump", "workability"],
    "references": ["is code", "bis", "standard", "specification", "reference", "doc", "code",
                   "is 456", "is 383", "is 9103"],
    "personnel": ["personnel", "engineer", "supervisor", "inspector", "technician", "responsible",
                  "site engineer", "quality control", "foreman"],
    "health_safety": ["safety", "ppe", "hazard", "protection", "health", "environment", "hse",
                      "weather", "frost", "rain", "temperature"],
    "acronyms": ["acronym", "abbreviation", "definition", "term", "terminology"],
    "scope": ["scope", "coverage", "applies", "applicable", "applicability", "covers"],
}


def classify_section_enhanced(title: str, content: str, chunk_type: str,
                             hierarchy_path: str) -> str:
    """
    Enhanced classification using:
    - Content keywords
    - Chunk type (section_content vs table)
    - Hierarchy path context
    - Metadata from parser
    """
    text = (title + " " + content).lower()
    
    # Combine title, content, and hierarchy for context
    full_context = f"{hierarchy_path} {title} {content}".lower()
    
    # Score each category
    scores = {}
    for category, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in full_context)
        scores[category] = score
    
    # Handle table chunks - they belong to their section's category
    if chunk_type == "table":
        # Look for strongest category signal
        best_category = max(scores, key=scores.get)
        if scores[best_category] > 0:
            return best_category
        # Default table to procedure if unsure (most tables are about methods)
        return "procedure"
    
    # Return category with highest score, or default to "other"
    best_category = max(scores, key=scores.get)
    if scores[best_category] > 0:
        return best_category
    
    return "other"


# ---------------------------------------------------------------------------
# Metadata Enrichment Helpers
# ---------------------------------------------------------------------------

def enrich_chunk_metadata(chunk: dict) -> dict:
    """
    Enrich chunk with additional metadata for better retrieval:
    - Content density (words per 100 chars)
    - Hierarchy depth level
    - Semantic similarity signals
    - Chunk position in sequence
    """
    metadata = chunk.get("metadata", {})
    content_length = metadata.get("content_length", 0)
    word_count = metadata.get("word_count", 0)
    
    # Calculate content density
    content_density = (word_count / max(content_length / 100, 1)) if content_length > 0 else 0
    
    # Determine content importance based on depth and size
    importance = 1.0
    if metadata.get("chunk_type") == "section_content":
        # Higher importance for top-level sections
        depth = metadata.get("depth", 0)
        importance = max(0.5, 1.0 / (1 + depth * 0.2))
    elif metadata.get("chunk_type") == "table":
        # Tables are important for factual retrieval
        importance = 1.2
    
    enriched = {
        **metadata,
        "content_density": round(content_density, 2),
        "importance_score": round(importance, 2),
        "is_continuation_chunk": metadata.get("is_continuation", False),
        "chunk_position": f"{metadata.get('chunk_index', 0)}/{metadata.get('total_chunks', 1)}",
    }
    
    return enriched


def build_embedding_text(chunk: dict) -> str:
    """
    Build optimized text for embedding that includes context:
    - Title (high weight in embedding)
    - Section number and hierarchy
    - Content
    - Table description (if table chunk)
    """
    parts = []
    metadata = chunk.get("metadata", {})
    
    # Add hierarchy context for better semantic understanding
    hierarchy = metadata.get("hierarchy_path", "")
    if hierarchy:
        parts.append(f"[Section: {hierarchy}]")
    
    # Add title with emphasis
    title = chunk.get("title", "")
    if title:
        parts.append(f"Title: {title}")
    
    # Add main text
    text = chunk.get("text", "")
    if text:
        parts.append(text)
    
    # For tables, add table description for semantic clarity
    if metadata.get("chunk_type") == "table":
        table_desc = metadata.get("table_description", "")
        if table_desc:
            parts.append(f"[Table Info: {table_desc}]")
    
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Backward Compatibility: Convert V2 format to V3
# ---------------------------------------------------------------------------

def convert_v2_to_v3_format(sections: list[dict]) -> list[dict]:
    """
    Convert old v2 format (section, title, content, tables, parent)
    to new v3 format with rich metadata.
    """
    chunks = []
    counter = 0
    
    for section in sections:
        section_id = section.get("section", "")
        title = section.get("title", "")
        content = section.get("content", "")
        parent = section.get("parent")
        tables = section.get("tables", [])
        
        # Generate hierarchy path (simplified for v2 data)
        hierarchy_parts = section_id.split(".")
        hierarchy_path = " > ".join(hierarchy_parts)
        
        # Calculate metadata
        content_length = len(content)
        word_count = len(content.split()) if content else 0
        sentence_count = len(re.split(r'[.!?]+', content)) if content else 0
        
        # Create section content chunk
        if content.strip():
            chunk = {
                "id": f"{section_id}_text",
                "section": section_id,
                "title": title,
                "text": f"{title}\n{content}" if content else title,
                "metadata": {
                    "section_id": section_id,
                    "parent": parent,
                    "hierarchy_path": hierarchy_path,
                    "depth": len(hierarchy_parts) - 1,
                    "chunk_type": "section_content",
                    "content_length": content_length,
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "has_tables": len(tables) > 0,
                    "num_tables": len(tables),
                }
            }
            chunks.append(chunk)
        
        # Create table chunks
        for tbl_idx, table in enumerate(tables):
            table_id = table.get("table_id", f"table_{tbl_idx}")
            table_data = table.get("data", [])
            
            # Build table text
            table_lines = [table_id]
            for row in table_data[:5]:
                row_str = " | ".join(str(c).strip() for c in row if c)
                if row_str.strip():
                    table_lines.append(row_str)
            if len(table_data) > 5:
                table_lines.append(f"... ({len(table_data) - 5} more rows)")
            
            table_text = "\n".join(table_lines)
            contextualised = f"{title}\n{table_text}"
            
            chunk = {
                "id": f"{section_id}_table_{tbl_idx}",
                "section": section_id,
                "title": title,
                "text": contextualised,
                "table_data": table_data,
                "metadata": {
                    "section_id": section_id,
                    "parent": parent,
                    "hierarchy_path": hierarchy_path,
                    "depth": len(hierarchy_parts) - 1,
                    "chunk_type": "table",
                    "table_id": table_id,
                    "table_rows": len(table_data),
                    "table_cols": len(table_data[0]) if table_data else 0,
                    "has_tables": True,
                    "num_tables": len(tables),
                }
            }
            chunks.append(chunk)
    
    return chunks


# ---------------------------------------------------------------------------
# Smart Chunk Processing
# ---------------------------------------------------------------------------

def ensure_unique_ids(chunks: list[dict]) -> list[dict]:
    """
    Ensure all chunk IDs are unique.
    If duplicates are found, append counter suffix.
    """
    seen_ids = {}
    processed_chunks = []
    
    for chunk in chunks:
        original_id = chunk.get("id", "chunk")
        
        if original_id in seen_ids:
            # Duplicate found - append counter
            counter = seen_ids[original_id]
            new_id = f"{original_id}_{counter}"
            seen_ids[original_id] += 1
        else:
            # First occurrence
            new_id = original_id
            seen_ids[original_id] = 1
        
        # Update chunk ID
        chunk["id"] = new_id
        processed_chunks.append(chunk)
    
    return processed_chunks


def is_v3_format(data: list[dict]) -> bool:
    """Detect if data is v3 format (has rich metadata structure)."""
    if not data or len(data) == 0:
        return False
    
    first_item = data[0]
    # V3 has 'id', 'metadata' with 'chunk_type', 'hierarchy_path'
    has_v3_markers = (
        "id" in first_item and
        "metadata" in first_item and
        "chunk_type" in first_item.get("metadata", {})
    )
    return has_v3_markers


# ---------------------------------------------------------------------------
# Build Chunks (V3 format - handles both v3 and v2 inputs)
# ---------------------------------------------------------------------------

def build_chunks(sections: list[dict]) -> list[dict]:
    """
    Process chunks for embedding and storage:
    1. Detect format (v3 with metadata or v2 plain sections)
    2. Convert v2 to v3 if needed
    3. Enrich metadata
    4. Add classification labels
    5. Ensure unique IDs
    6. Prepare for embedding
    """
    
    # Auto-detect and convert format if needed
    if is_v3_format(sections):
        print("📦 Detected v3 format (with semantic chunking and metadata)")
        chunks = sections
    else:
        print("📦 Detected v2 format - converting to v3...")
        chunks = convert_v2_to_v3_format(sections)
    
    # Process and enrich each chunk
    enriched_chunks = []
    
    for chunk in chunks:
        # Get metadata
        metadata = chunk.get("metadata", {})
        
        # Add classification label based on enhanced algorithm
        label = classify_section_enhanced(
            title=chunk.get("title", ""),
            content=chunk.get("text", ""),
            chunk_type=metadata.get("chunk_type", "section_content"),
            hierarchy_path=metadata.get("hierarchy_path", "")
        )
        
        # Enrich with additional metadata
        enriched_metadata = enrich_chunk_metadata(chunk)
        enriched_metadata["label"] = label
        
        # Build optimized embedding text
        embedding_text = build_embedding_text(chunk)
        
        # Create enhanced chunk
        enriched_chunk = {
            "id": chunk.get("id", f"chunk_{len(enriched_chunks)}"),
            "text": embedding_text,  # Text for embedding
            "original_text": chunk.get("text", ""),  # Original text for retrieval
            "metadata": enriched_metadata,
        }
        
        enriched_chunks.append(enriched_chunk)
    
    # ---- Ensure all IDs are unique (handles semantic chunking duplicates) ----
    print(f"\n🔍 Deduplicating chunk IDs...")
    enriched_chunks = ensure_unique_ids(enriched_chunks)
    
    # Count how many IDs were modified
    id_collisions = sum(1 for c in enriched_chunks if "_" in c["id"].split("_")[-1] and c["id"].split("_")[-1].isdigit())
    if id_collisions > 0:
        print(f"⚠️  Fixed {id_collisions} duplicate IDs by appending counters")
    
    return enriched_chunks


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main(parsed_file: str, db_dir: str = "chroma_db", 
         embedding_model: str = "all-MiniLM-L12-v2"):
    """
    Main pipeline:
    1. Load parsed sections (v2 or v3 format)
    2. Build/enrich chunks with metadata
    3. Embed using SentenceTransformer
    4. Store in ChromaDB with rich metadata
    """
    
    print("\n" + "=" * 80)
    print("EMBED & STORE PIPELINE (v4 - Enhanced)")
    print("=" * 80)
    
    # ---- Load data ----
    print(f"\n📖 Loading parsed sections from: {parsed_file}")
    with open(parsed_file, "r", encoding="utf-8") as f:
        sections = json.load(f)
    print(f"✅ Loaded {len(sections)} sections/chunks")
    
    # ---- Build chunks ----
    print(f"\n🔧 Building and enriching chunks...")
    chunks = build_chunks(sections)
    print(f"✅ Built {len(chunks)} enhanced chunks")
    
    # ---- Statistics ----
    section_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'section_content']
    table_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'table']
    
    print(f"\n📊 Chunk Statistics:")
    print(f"   • Total Chunks       : {len(chunks)}")
    print(f"   • Section Chunks     : {len(section_chunks)}")
    print(f"   • Table Chunks       : {len(table_chunks)}")
    
    # Label distribution
    label_dist = {}
    for c in chunks:
        label = c['metadata'].get('label', 'unknown')
        label_dist[label] = label_dist.get(label, 0) + 1
    
    print(f"\n🏷️  Label Distribution:")
    for label in sorted(label_dist.keys()):
        count = label_dist[label]
        pct = (count / len(chunks)) * 100
        print(f"   • {label:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Depth distribution
    depth_dist = {}
    for c in chunks:
        depth = c['metadata'].get('depth', 0)
        depth_dist[depth] = depth_dist.get(depth, 0) + 1
    
    print(f"\n📍 Hierarchy Depth Distribution:")
    for depth in sorted(depth_dist.keys()):
        count = depth_dist[depth]
        print(f"   • Level {depth}: {count:4d} chunks")
    
    # Average content metrics
    avg_content_len = sum(c['metadata'].get('content_length', 0) for c in chunks) / max(len(chunks), 1)
    avg_word_count = sum(c['metadata'].get('word_count', 0) for c in chunks) / max(len(chunks), 1)
    
    print(f"\n📈 Content Metrics:")
    print(f"   • Avg Content Length : {avg_content_len:.0f} chars")
    print(f"   • Avg Word Count     : {avg_word_count:.0f} words")
    print(f"   • Continuation Chunks: {sum(1 for c in chunks if c['metadata'].get('is_continuation_chunk'))} chunks")
    
    # ---- Embed ----
    print(f"\n🧠 Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    print(f"✅ Model loaded (embedding dimension: {model.get_sentence_embedding_dimension()})")
    
    print(f"\n⚙️  Generating embeddings for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
    print(f"✅ Generated {len(embeddings_list)} embeddings")
    
    # ---- Store in ChromaDB ----
    print(f"\n💾 Connecting to ChromaDB at: {db_dir}")
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection("sections")
    print(f"✅ Connected to collection: sections")
    
    print(f"\n📝 Upserting chunks into ChromaDB...")
    
    # Verify no duplicate IDs before upsert
    chunk_ids = [c["id"] for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        duplicate_ids = [id for id in set(chunk_ids) if chunk_ids.count(id) > 1]
        print(f"❌ ERROR: Found duplicate IDs: {duplicate_ids}")
        print("   This should not happen. Please check your parsed_sections.json")
        sys.exit(1)
    
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings_list,
        documents=[c["original_text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"✅ Upserted {len(chunks)} chunks into ChromaDB")
    
    # ---- Summary ----
    print("\n" + "=" * 80)
    print("✅ EMBEDDING & STORAGE COMPLETE")
    print("=" * 80)
    print(f"\n📦 Summary:")
    print(f"   • Input Format       : {'v3 (semantic chunking)' if is_v3_format(sections) else 'v2 (legacy)'}")
    print(f"   • Total Chunks       : {len(chunks)}")
    print(f"   • Embedding Model    : {embedding_model}")
    print(f"   • Embedding Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   • Database Location  : ./{db_dir}")
    print(f"   • Collection         : sections")
    print("\n✨ Ready for retrieval and generation!")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embed_and_store.py <parsed_sections.json> [db_dir] [embedding_model]")
        print("")
        print("Arguments:")
        print("  parsed_sections.json : Path to parsed sections (v2 or v3 format)")
        print("  db_dir              : ChromaDB directory (default: chroma_db)")
        print("  embedding_model     : Model name (default: all-MiniLM-L12-v2)")
        print("")
        print("Example embedding models:")
        print("  - all-MiniLM-L6-v2  (lightweight, baseline)")
        print("  - all-MiniLM-L12-v2 (improved, recommended)")
        print("  - nomic-embed-text-v1.5 (domain-aware, experimental)")
        print("")
        print("⚠️  NOTE: If you modified the parser, delete chroma_db/ before running:")
        print("        rmdir /s /q chroma_db    (Windows)")
        print("        rm -rf chroma_db          (Linux/Mac)")
        sys.exit(1)

    parsed_file = sys.argv[1]
    db_dir = sys.argv[2] if len(sys.argv) > 2 else "chroma_db"
    embedding_model = sys.argv[3] if len(sys.argv) > 3 else "all-MiniLM-L12-v2"
    
    main(parsed_file, db_dir, embedding_model)