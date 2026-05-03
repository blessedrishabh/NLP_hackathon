"""
parser.py  –  CPWD Specification PDF Parser (v3)
=================================================
Improvements over v2:
- Semantic chunking: Long sections split with overlap for better context
- Rich metadata: hierarchy, depth, content stats, chunk type
- Enhanced tables: auto-generated descriptions, linked to parent sections
- Better text processing: cleaner validation, improved special char handling
- Chunk overlap: adjacent chunks merged when semantically related
- Hierarchical paths: full parent chain tracked for context
"""

import re
import sys
import json
import fitz                          # PyMuPDF
from typing import Optional

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
SECTION_REGEX   = re.compile(r'^(\d+(?:\.\d+)+)\.?\s+(.*)')
TABLE_CAP_REGEX = re.compile(r'(TABLE|Table)\s+(\d+\.\d+)\b(.*)', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Semantic Chunking Helpers
# ---------------------------------------------------------------------------
def get_hierarchy_path(node: 'SectionNode', parent_map: dict) -> str:
    """Build full hierarchical path: e.g., '5 > 5.1 > 5.1.2'"""
    path = [node.section_id]
    current = node
    while current.section_id in parent_map:
        current = parent_map[current.section_id]
        if current.section_id != "root":
            path.insert(0, current.section_id)
    return " > ".join(path)


def split_long_content(content: str, max_chunk_size: int = 800, 
                       overlap_size: int = 150) -> list[str]:
    """
    Split long content into semantic chunks with overlap.
    Tries to split on sentence boundaries when possible.
    """
    if len(content) <= max_chunk_size:
        return [content]
    
    sentences = re.split(r'(?<=[.!?])\s+', content)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Add overlap between chunks
    if len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-overlap_size:] if len(prev_chunk) > overlap_size else prev_chunk
            overlapped.append(overlap_text + " " + chunks[i])
        return overlapped
    
    return chunks


def generate_table_description(table_id: str, data: list[list[str]]) -> str:
    """Generate semantic description of table content."""
    if not data or len(data) == 0:
        return f"{table_id}: (empty)"
    
    description = f"{table_id}:"
    
    # Extract header if available
    if len(data) > 0:
        header = [h for h in data[0] if h.strip()]
        if header:
            description += f" Columns: {', '.join(header[:3])}"
            if len(header) > 3:
                description += f" (+{len(header) - 3} more)"
    
    # Count non-empty rows
    non_empty_rows = sum(1 for row in data if any(c.strip() for c in row))
    description += f". Rows: ~{non_empty_rows}"
    
    return description


# ---------------------------------------------------------------------------
# Section tree
# ---------------------------------------------------------------------------
class SectionNode:
    def __init__(self, section_id: str, title: str):
        self.section_id = section_id
        self.title      = title
        self.content    = []   # list[str]  – plain text fragments
        self.tables     = []   # list[dict] – structured tables
        self.children   = []   # list[SectionNode]

    def add_content(self, text: str):
        self.content.append(text)

    def add_table(self, table: dict):
        self.tables.append(table)

    def to_dict(self) -> dict:
        return {
            "section" : self.section_id,
            "title"   : self.title,
            "content" : " ".join(self.content).strip(),
            "tables"  : self.tables,
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# Section-level helpers (unchanged from v1)
# ---------------------------------------------------------------------------
def get_level(section_id: str) -> int:
    parts = section_id.split(".")
    if parts[-1] == "0":
        return len(parts) - 2   # "4.0" → level 0
    return len(parts) - 1       # "4.1" → level 1


def find_parent(stack: list, section_id: str) -> SectionNode:
    current_level = get_level(section_id)
    for node in reversed(stack):
        if node.section_id == "root":
            return node
        if get_level(node.section_id) == current_level - 1:
            return node
    return stack[0]


def is_valid_section(text: str) -> bool:
    """Enhanced validation to filter out noise and metadata lines."""
    text = text.strip()
    
    # Filter out pure measurement/units
    if re.search(r'^\s*[\d\.\-\s()]+\s*(mm|cm|m|kg|%|mm²|m³|N\/mm²)\s*$', text.lower()):
        return False
    
    # Filter out lines that are mostly digits (page numbers, indices, etc.)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.4:
        return False
    
    # Filter out pure numeric/symbol patterns
    if re.match(r'^[\d\.\-\s()]+$', text):
        return False
    
    # Must contain at least some alphabetic characters
    if not re.search(r'[A-Za-z]', text):
        return False
    
    # Minimum length requirement
    if len(text) < 5:
        return False
    
    # Filter out common header/footer patterns
    if re.match(r'^(page|chapter|table of contents|index|appendix|glossary)', text.lower()):
        return False
    
    return True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def rect_contains(outer, inner_point) -> bool:
    """True if (x, y) falls inside fitz.Rect / bbox tuple."""
    x0, y0, x1, y1 = outer
    px, py = inner_point
    # small tolerance so border lines don't get swallowed
    return x0 - 2 < px < x1 + 2 and y0 - 2 < py < y1 + 2


def point_in_any_table(px, py, table_bboxes) -> bool:
    return any(rect_contains(bb, (px, py)) for bb in table_bboxes)


# ---------------------------------------------------------------------------
# Table caption detection
# ---------------------------------------------------------------------------
def clean_table_rows(raw_rows: list[list]) -> list[list[str]]:
    """
    Convert a PyMuPDF table (list of lists of str|None) into clean strings.
    Merges multi-line cell text (\\n → space) and replaces None with ''.
    """
    cleaned = []
    for row in raw_rows:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                cleaned_row.append(str(cell).replace("\n", " ").strip())
        cleaned.append(cleaned_row)
    return cleaned


def extract_embedded_caption(rows: list[list[str]]) -> str | None:
    """
    Some CPWD tables embed their title in a merged top cell, e.g.
    first row = ['TABLE 4.1 Graded Stone Aggregate or Gravel', '', '', …]
    Return the caption string if found, else None.
    """
    if not rows:
        return None
    first_non_empty = [c for c in rows[0] if c]
    if not first_non_empty:
        return None
    candidate = first_non_empty[0]
    m = TABLE_CAP_REGEX.match(candidate)
    if m:
        label = f"Table {m.group(2)}"
        suffix = m.group(3).strip(" :-")
        return f"{label} {suffix}".strip() if suffix else label
    return None


def drop_embedded_caption_row(rows: list[list[str]]) -> list[list[str]]:
    """Remove the first row if it was the embedded caption row."""
    if not rows:
        return rows
    first_non_empty = [c for c in rows[0] if c]
    if first_non_empty and TABLE_CAP_REGEX.match(first_non_empty[0]):
        return rows[1:]
    return rows


# ---------------------------------------------------------------------------
# Core extraction pass
# ---------------------------------------------------------------------------
def extract_page_items(doc):
    """
    Walk every page and return a flat list of items:
        {"type": "text",  "text": str, "y": float, "page": int}
        {"type": "table", "table_id": str, "data": list[list[str]],
                          "y": float, "page": int}

    Items are ordered by (page, y-coordinate) so downstream code sees them
    in reading order.
    """
    items = []

    for page_num, page in enumerate(doc):
        # ---- 1. Detect tables on this page --------------------------------
        tab_finder   = page.find_tables()
        page_tables  = tab_finder.tables          # list of fitz.table objects
        table_bboxes = [t.bbox for t in page_tables]

        # Pending caption from a preceding "TABLE X.X …" text line
        pending_caption: str | None = None

        # ---- 2. Collect text blocks (skip those inside table regions) -----
        text_data  = page.get_text("dict")
        text_items = []  # (y_top, text_str)

        for block in text_data["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text  = ""
                y_center   = (line["bbox"][1] + line["bbox"][3]) / 2
                x_left     = line["bbox"][0]

                for span in line["spans"]:
                    line_text += span["text"]
                line_text = line_text.strip()
                if not line_text:
                    continue

                # Suppress text that is visually inside a table bbox
                if point_in_any_table(x_left, y_center, table_bboxes):
                    continue

                text_items.append((line["bbox"][1], line_text))

        # ---- 3. Slot tables into reading-order alongside text -------------
        # Build [(y, item_dict)] for text
        ordered: list[tuple[float, dict]] = []
        for y, txt in text_items:
            ordered.append((y, {"type": "text", "text": txt,
                                 "y": y, "page": page_num}))

        # Add tables keyed by their top-y
        for tbl in page_tables:
            y_top  = tbl.bbox[1]
            rows   = clean_table_rows(tbl.extract())
            cap    = extract_embedded_caption(rows)
            if cap:
                rows = drop_embedded_caption_row(rows)
            # table_id will be resolved below (from preceding caption text)
            ordered.append((y_top, {"type": "table_pending",
                                    "embedded_cap": cap,
                                    "data": rows,
                                    "y": y_top,
                                    "page": page_num}))

        ordered.sort(key=lambda x: x[0])

        # ---- 4. Resolve table captions from text flow ---------------------
        for _, item in ordered:
            if item["type"] == "text":
                m = TABLE_CAP_REGEX.match(item["text"])
                if m:
                    label  = f"Table {m.group(2)}"
                    suffix = m.group(3).strip(" :-")
                    pending_caption = f"{label} {suffix}".strip() if suffix else label
                    # Don't emit the caption line as plain text
                    continue
                items.append(item)

            elif item["type"] == "table_pending":
                # Prefer explicit caption text over embedded one
                table_id = pending_caption or item["embedded_cap"] or "Table (unknown)"
                pending_caption = None
                items.append({
                    "type"    : "table",
                    "table_id": table_id,
                    "data"    : item["data"],
                    "y"       : item["y"],
                    "page"    : item["page"],
                })

    return items


# ---------------------------------------------------------------------------
# Merge tables that were split across a page boundary
# ---------------------------------------------------------------------------
def merge_split_tables(items: list[dict]) -> list[dict]:
    """
    When a table is cut by a page break PyMuPDF returns two separate table
    objects.  The continuation has no caption ('Table (unknown)') and sits at
    the very top of the next page.  Heuristic: if consecutive table items share
    the same column count AND the second one has no real caption, merge them.
    """
    merged: list[dict] = []
    i = 0
    while i < len(items):
        item = items[i]
        if item["type"] == "table" and i + 1 < len(items):
            nxt = items[i + 1]
            # Look ahead: skip any "text" items between them (page headers etc.)
            j = i + 1
            while j < len(items) and items[j]["type"] == "text":
                j += 1
            if j < len(items):
                nxt = items[j]
                is_continuation = (
                    nxt["type"] == "table"
                    and "unknown" in nxt["table_id"].lower()
                    and nxt["page"] == item["page"] + 1   # next page only
                    and len(nxt["data"]) > 0
                    and len(item["data"]) > 0
                    and len(nxt["data"][0]) == len(item["data"][0])  # same cols
                )
                if is_continuation:
                    # Merge rows and drop the look-ahead item
                    item = dict(item)
                    item["data"] = item["data"] + nxt["data"]
                    # Also keep the skipped text items
                    merged.append(item)
                    merged.extend(items[i + 1:j])  # text items between tables
                    i = j + 1
                    continue
        merged.append(item)
        i += 1
    return merged


# ---------------------------------------------------------------------------
# Build section tree from item stream
# ---------------------------------------------------------------------------
def build_tree(items: list[dict]) -> SectionNode:
    root         = SectionNode("root", "Document")
    stack        = [root]
    current_node = root

    for item in items:
        if item["type"] == "text":
            text  = item["text"]
            match = SECTION_REGEX.match(text)

            if match:
                section_id = match.group(1)
                rest       = match.group(2).strip().lstrip(".").strip()

                if ":" in rest:
                    title, remaining = (p.strip() for p in rest.split(":", 1))
                else:
                    title, remaining = rest.strip(), ""

                title = title.strip(" .:-")

                if title and is_valid_section(title):
                    level    = get_level(section_id)
                    new_node = SectionNode(section_id, title)
                    if remaining:
                        new_node.add_content(remaining)

                    parent = find_parent(stack, section_id)
                    parent.children.append(new_node)

                    stack = [n for n in stack
                             if n.section_id == "root"
                             or get_level(n.section_id) < level]
                    stack.append(new_node)
                    current_node = new_node
                    continue

            current_node.add_content(text)

        elif item["type"] == "table":
            current_node.add_table({
                "table_id": item["table_id"],
                "data"    : item["data"],
            })

    return root


# ---------------------------------------------------------------------------
# Flatten tree to list of section dicts with semantic chunking
# ---------------------------------------------------------------------------
def flatten_sections(node: SectionNode, parent: SectionNode | None = None,
                     parent_map: dict | None = None,
                     depth: int = 0) -> list[dict]:
    """
    Flatten section tree with:
    - Semantic chunking for long content
    - Rich metadata (hierarchy, depth, stats, chunk type)
    - Table descriptions and linking
    - Parent-child relationships preserved
    """
    if parent_map is None:
        parent_map = {}
    
    chunks = []
    
    if node.section_id != "root":
        # Store parent mapping for hierarchy construction
        if parent:
            parent_map[node.section_id] = parent
        
        base_metadata = {
            "section_id": node.section_id,
            "parent": parent.section_id if parent else None,
            "hierarchy_path": get_hierarchy_path(node, parent_map),
            "depth": depth,
            "title": node.title,
            "has_tables": len(node.tables) > 0,
            "num_tables": len(node.tables),
        }
        
        # ---- 1. Create section chunk with tables ----
        content_text = " ".join(node.content).strip()
        section_text = f"{node.title}\n{content_text}" if content_text else node.title
        
        # Content statistics
        content_stats = {
            "content_length": len(content_text),
            "word_count": len(content_text.split()),
            "sentence_count": len(re.split(r'[.!?]+', content_text)),
        }
        
        if section_text.strip():
            # Semantic chunk splitting for long content
            if len(content_text) > 800:
                content_chunks = split_long_content(content_text, max_chunk_size=800, overlap_size=150)
            else:
                content_chunks = [content_text]
            
            for chunk_idx, content_chunk in enumerate(content_chunks):
                chunk_dict = {
                    "id": f"{node.section_id}_text_{chunk_idx}" if len(content_chunks) > 1 else f"{node.section_id}_text",
                    "section": node.section_id,
                    "title": node.title,
                    "text": content_chunk,
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "section_content",
                        "is_continuation": chunk_idx > 0,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(content_chunks),
                        **content_stats,
                        "table_count_in_section": len(node.tables),
                    }
                }
                chunks.append(chunk_dict)
        
        # ---- 2. Create table chunks with context ----
        for table_idx, table in enumerate(node.tables):
            table_id = table.get("table_id", f"table_{table_idx}")
            table_data = table.get("data", [])
            
            # Generate semantic description
            table_desc = generate_table_description(table_id, table_data)
            
            # Create table text with context
            table_text = f"{node.title}\n{table_desc}\n"
            
            # Add first few rows as preview
            preview_rows = table_data[:5] if len(table_data) > 0 else []
            for row in preview_rows:
                row_str = " | ".join(str(cell).strip() for cell in row if cell)
                if row_str.strip():
                    table_text += f"{row_str}\n"
            
            if len(table_data) > 5:
                table_text += f"... ({len(table_data) - 5} more rows)"
            
            table_metadata = {
                **base_metadata,
                "chunk_type": "table",
                "table_id": table_id,
                "table_index": table_idx,
                "total_tables_in_section": len(node.tables),
                "table_rows": len(table_data),
                "table_cols": len(table_data[0]) if len(table_data) > 0 else 0,
                "table_description": table_desc,
                **content_stats,
            }
            
            chunks.append({
                "id": f"{node.section_id}_table_{table_idx}",
                "section": node.section_id,
                "title": node.title,
                "text": table_text,
                "table_data": table_data,  # For later reference
                "metadata": table_metadata,
            })
    
    # Recursively process children with increased depth
    for child in node.children:
        chunks.extend(flatten_sections(child, node, parent_map, depth + 1))
    
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse(pdf_path: str) -> list[dict]:
    doc   = fitz.open(pdf_path)
    items = extract_page_items(doc)
    items = merge_split_tables(items)
    tree  = build_tree(items)
    return flatten_sections(tree)


def convert_new_to_old_format(enhanced_chunks: list[dict]) -> list[dict]:
    """
    Convert v3 enhanced chunks back to v2 format for backward compatibility.
    Groups chunks by section and reconstructs the old structure.
    """
    sections_map = {}
    
    for chunk in enhanced_chunks:
        section_id = chunk.get('section')
        if not section_id:
            continue
            
        if section_id not in sections_map:
            sections_map[section_id] = {
                "section": section_id,
                "title": chunk.get('title', ''),
                "content": "",
                "tables": [],
                "parent": chunk['metadata'].get('parent'),
            }
        
        # Accumulate content
        if chunk['metadata'].get('chunk_type') == 'section_content':
            if sections_map[section_id]["content"]:
                sections_map[section_id]["content"] += " "
            sections_map[section_id]["content"] += chunk.get('text', '').replace(chunk.get('title', ''), '').strip()
        
        # Collect tables
        elif chunk['metadata'].get('chunk_type') == 'table':
            if 'table_data' in chunk:
                sections_map[section_id]["tables"].append({
                    "table_id": chunk['metadata'].get('table_id', ''),
                    "data": chunk['table_data'],
                })
    
    return list(sections_map.values())


if __name__ == "__main__":
    pdf_path = (sys.argv[1] if len(sys.argv) > 1
                else "Prescriptive Specifications_CPWD.pdf")

    chunks = parse(pdf_path)

    # ---- pretty-print first few enhanced chunks ----------------------------------
    print("\n" + "=" * 80)
    print("ENHANCED PARSER OUTPUT (v3 - Semantic Chunking & Rich Metadata)")
    print("=" * 80)
    
    for idx, c in enumerate(chunks[:8]):
        print(f"\n[CHUNK {idx}]")
        print(f"  ID              : {c.get('id', 'N/A')}")
        print(f"  Section         : {c.get('section', 'N/A')}")
        print(f"  Title           : {c.get('title', 'N/A')}")
        print(f"  Chunk Type      : {c['metadata'].get('chunk_type', 'N/A')}")
        print(f"  Hierarchy Path  : {c['metadata'].get('hierarchy_path', 'N/A')}")
        print(f"  Depth           : {c['metadata'].get('depth', 'N/A')}")
        print(f"  Content Length  : {c['metadata'].get('content_length', 'N/A')} chars")
        print(f"  Word Count      : {c['metadata'].get('word_count', 'N/A')}")
        
        if c['metadata'].get('chunk_type') == 'table':
            print(f"  Table Desc      : {c['metadata'].get('table_description', 'N/A')}")
            print(f"  Table Dims      : {c['metadata'].get('table_rows', 0)} rows × {c['metadata'].get('table_cols', 0)} cols")
        
        text_preview = c.get('text', '')[:150].replace('\n', ' ')
        print(f"  Text Preview    : {text_preview}...")

    # ---- Save enhanced chunks ----
    with open("parsed_sections.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✅ Saved {len(chunks)} enhanced chunks → parsed_sections.json")
    print("=" * 80)
    
    # ---- Compute and display statistics ----
    section_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'section_content']
    table_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'table']
    
    total_content_length = sum(c['metadata'].get('content_length', 0) for c in section_chunks)
    total_tables = len(table_chunks)
    
    print(f"\nStatistics:")
    print(f"  Total Chunks       : {len(chunks)}")
    print(f"  Section Chunks     : {len(section_chunks)}")
    print(f"  Table Chunks       : {total_tables}")
    print(f"  Total Content Size : {total_content_length:,} characters")
    print(f"  Avg Chunk Size     : {total_content_length // max(len(section_chunks), 1):,} chars")
    
    # ---- Show hierarchy depth distribution ----
    depth_dist = {}
    for c in chunks:
        d = c['metadata'].get('depth', 0)
        depth_dist[d] = depth_dist.get(d, 0) + 1
    print(f"\nDepth Distribution:")
    for depth in sorted(depth_dist.keys()):
        print(f"  Level {depth}: {depth_dist[depth]} chunks")