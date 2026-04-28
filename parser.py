"""
parser.py  –  CPWD Specification PDF Parser (v2)
=================================================
Improvements over v1:
- Tables are extracted as structured 2-D arrays instead of raw text
- Each section carries a `tables` list alongside its `content` string
- Table captions ("TABLE 4.1 …") are resolved from surrounding text
- Text that falls inside a detected table bounding-box is suppressed so it
  does not bleed into the `content` field
"""

import re
import sys
import json
import fitz                          # PyMuPDF

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
SECTION_REGEX   = re.compile(r'^(\d+(?:\.\d+)+)\.?\s+(.*)')
TABLE_CAP_REGEX = re.compile(r'(TABLE|Table)\s+(\d+\.\d+)\b(.*)', re.IGNORECASE)


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
    text = text.strip()
    if re.search(r'\b(mm|cm|m|kg|%)\b', text.lower()):
        return False
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.4:
        return False
    if re.match(r'^[\d\.\-\s()]+$', text):
        return False
    if not re.search(r'[A-Za-z]', text):
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
# Flatten tree to list of section dicts
# ---------------------------------------------------------------------------
def flatten_sections(node: SectionNode, parent: SectionNode | None = None) -> list[dict]:
    chunks = []
    if node.section_id != "root":
        chunks.append({
            "section": node.section_id,
            "title"  : node.title,
            "content": " ".join(node.content).strip(),
            "tables" : node.tables,
            "parent" : parent.section_id if parent else None,
        })
    for child in node.children:
        chunks.extend(flatten_sections(child, node))
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


if __name__ == "__main__":
    pdf_path = (sys.argv[1] if len(sys.argv) > 1
                else "Prescriptive Specifications_CPWD.pdf")

    chunks = parse(pdf_path)

    # ---- pretty-print first few sections ----------------------------------
    for c in chunks[:12]:
        print("=" * 60)
        print(f"Section : {c['section']}")
        print(f"Title   : {c['title']}")
        print(f"Parent  : {c['parent']}")
        print(f"Content : {c['content'][:200]}")
        if c["tables"]:
            for t in c["tables"]:
                print(f"  [TABLE] {t['table_id']}")
                for row in t["data"][:3]:
                    print(f"         {row}")

    with open("parsed_sections.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("\nSaved → parsed_sections.json")