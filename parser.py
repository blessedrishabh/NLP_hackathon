import fitz  # PyMuPDF
import re
from collections import defaultdict
import json

SECTION_REGEX = re.compile(r'^(\d+(?:\.\d+)+)\.?\s+(.*)')


class SectionNode:
    def __init__(self, section_id, title):
        self.section_id = section_id
        self.title = title
        self.content = []
        self.children = []

    def add_content(self, text):
        self.content.append(text)

    def to_dict(self):
        return {
            "section": self.section_id,
            "title": self.title,
            "content": " ".join(self.content).strip(),
            "children": [child.to_dict() for child in self.children]
        }


def get_level(section_id):
    """
    BUG FIX: sections like '4.0' were being assigned level 1 (one dot),
    same as '4.1', making them siblings instead of parent/child.

    Fix: sections whose last component is '0' are treated as chapter-level
    (level 0), regardless of how many dots they contain.
    Examples:
        '4.0'     -> level 0  (chapter heading)
        '4.1'     -> level 1
        '4.1.1'   -> level 2
        '4.1.1.1' -> level 3
    """
    parts = section_id.split(".")
    if parts[-1] == "0":
        # Treat X.0 as one level above X.1 by stripping the trailing .0
        return len(parts) - 2  # e.g. "4.0" → 2 parts → level 0
    return len(parts) - 1      # e.g. "4.1" → 2 parts → level 1


def find_parent(stack, section_id):
    """
    BUG FIX: the old code pruned the stack BEFORE calling find_parent,
    which could remove the intended parent before we could look it up.

    Now the stack is pruned AFTER find_parent succeeds (see main loop).
    This function simply walks backwards to find the nearest ancestor
    whose level is exactly current_level - 1.
    """
    current_level = get_level(section_id)

    for node in reversed(stack):
        if node.section_id == "root":
            return node
        node_level = get_level(node.section_id)
        if node_level == current_level - 1:
            return node

    return stack[0]  # fallback to root


def is_valid_section(text):
    text = text.strip()

    # Reject measurement units
    if re.search(r'\b(mm|cm|m|kg|%)\b', text.lower()):
        return False

    # Reject numeric-heavy rows (tables)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.4:
        return False

    # Reject pure numeric patterns
    if re.match(r'^[\d\.\-\s()]+$', text):
        return False

    # Must contain alphabets (real title)
    if not re.search(r'[A-Za-z]', text):
        return False

    return True


def extract_blocks_with_font(doc):
    """Extract text with font size info"""
    blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        data = page.get_text("dict")
        for block in data["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                font_sizes = []
                for span in line["spans"]:
                    line_text += span["text"]
                    font_sizes.append(span["size"])
                if line_text.strip():
                    avg_font = sum(font_sizes) / len(font_sizes)
                    blocks.append({
                        "text": line_text.strip(),
                        "font_size": avg_font
                    })
    return blocks


def parse_with_font_detection(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = extract_blocks_with_font(doc)

    root = SectionNode("root", "Document")
    stack = [root]
    current_node = root

    for b in blocks:
        text = b["text"]
        match = SECTION_REGEX.match(text)

        if match:
            section_id = match.group(1)
            rest = match.group(2).strip().lstrip(".").strip()

            # Split title from inline content early, so we only validate
            # the title against unit/numeric checks — not downstream content.
            # e.g. "4.1.1.1 General: ...4.75 mm IS Sieve..." should pass
            # because the title "General" contains no units.
            if ":" in rest:
                parts = rest.split(":", 1)
                title = parts[0].strip()
                remaining = parts[1].strip()
            else:
                title = rest.strip()
                remaining = ""

            title = title.strip(" .:-")
            is_heading = bool(title) and is_valid_section(title)
        else:
            is_heading = False
            title = remaining = section_id = ""

        if is_heading:
            level = get_level(section_id)
            new_node = SectionNode(section_id, title)
            if remaining:
                new_node.add_content(remaining)

            # BUG FIX: find parent BEFORE pruning the stack
            parent = find_parent(stack, section_id)
            parent.children.append(new_node)

            # Now prune: drop anything at same level or deeper
            stack = [n for n in stack if n.section_id == "root" or get_level(n.section_id) < level]
            stack.append(new_node)
            current_node = new_node

        else:
            current_node.add_content(text)

    return root


def flatten_sections(node, parent=None):
    chunks = []
    if node.section_id != "root":
        chunks.append({
            "section": node.section_id,
            "title": node.title,
            "content": " ".join(node.content),
            "parent": parent.section_id if parent else None
        })
    for child in node.children:
        chunks.extend(flatten_sections(child, node))
    return chunks


# ===== RUN =====
if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "D:/NLP_Hackathon/project/Prescriptive Specifications_CPWD.pdf"

    tree = parse_with_font_detection(pdf_path)
    chunks = flatten_sections(tree)

    for c in chunks[:10]:
        print("=" * 50)
        print("Section:", c["section"])
        print("Title  :", c["title"])
        print("Parent :", c["parent"])
        print("Content:", c["content"][:200])
    with open("parsed_sections.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Saved parsed sections to parsed_sections.json")