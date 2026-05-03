# Parser.py - Video Explanation Guide

## 🎯 Project Strategy Overview

Your **parser.py** is a sophisticated **semantic PDF extraction engine** designed specifically for CPWD (Central Public Works Department) construction specification documents. It's a **rule-based, hierarchical chunking system** with NO machine learning models—purely intelligent text processing.

---

## 📚 Core Strategy

### **What Problem Does It Solve?**
- PDFs contain mixed content: hierarchical sections, tables, and unstructured text
- Simply extracting raw text loses important context and structure
- Goal: Convert PDF → **semantically meaningful, hierarchical chunks** with rich metadata

### **Three-Phase Approach**

```
Phase 1: Extraction         Phase 2: Hierarchization        Phase 3: Chunking & Enrichment
─────────────────────────   ──────────────────────────      ─────────────────────────────
PDF → PyMuPDF Extract       Build Section Tree              Semantic Splitting
Text + Tables + Position    (1.0, 1.1, 1.2.1, etc.)        Metadata Enrichment
Detection Order Preserved   Parent-Child Links              Overlap Addition
```

---

## 🔧 Modules & Libraries Used

### **PyMuPDF (fitz)**
```python
import fitz  # PDF extraction library
```
- **What it does:** Extracts text, tables, and layout information from PDFs
- **Key methods used:**
  - `fitz.open(pdf_path)` - Opens PDF document
  - `page.find_tables()` - Detects table regions and structure
  - `page.get_text("dict")` - Extracts text with bounding box coordinates
  - `page.extract()` - Extracts full page content

### **Standard Python Libraries**
- **`re` (regex):** Pattern matching for section headers and table captions
- **`json`:** Serialization of structured output
- **`sys`:** Command-line argument handling

### **No ML Models!**
- Pure rule-based extraction
- No neural networks, no embeddings in the parser
- Downstream RAG system uses embeddings, NOT this parser

---

## 📋 Extraction Method (How It Works)

### **Step 1: Page-by-Page Scanning**
```python
for page_num, page in enumerate(doc):
    # 1. Find tables on page
    # 2. Extract text blocks with coordinates
    # 3. Maintain reading order (top-to-bottom, left-to-right)
```

**Key Point:** Uses **Y-coordinates** to preserve reading order

### **Step 2: Intelligent Text vs. Table Separation**
```
[Geometric Check]
  ↓
Is text inside a table bbox?
  └─ YES → Skip (avoid duplication)
  └─ NO  → Add to text items
```

### **Step 3: Table Detection & Caption Linking**
```
Scenario 1: Explicit Caption
  "TABLE 4.1 Graded Stone Aggregate"
         ↓
  [regex match: TABLE \d+\.\d+]
         ↓
  Link to next table found

Scenario 2: Embedded Caption
  Table cell: ["TABLE 4.1 Description", "", ...]
         ↓
  Extract from first row
         ↓
  Remove from table data

Scenario 3: Cross-Page Table
  Table split at page boundary?
         ↓
  [Heuristic: same columns + "unknown" ID + consecutive pages]
         ↓
  Merge all rows into one table
```

### **Step 4: Build Hierarchical Tree**

```
Document (root)
├─ 1. Materials
│  ├─ 1.1. Cement
│  │  ├─ 1.1.1. Portland Cement
│  │  └─ 1.1.2. Blended Cement
│  └─ 1.2. Aggregates
│     └─ [Tables for aggregate specifications]
├─ 2. Methodology
│  └─ [Sections and content]
└─ 3. Quality Control
   └─ [Tables with quality metrics]
```

**Section ID Pattern:** `\d+(?:\.\d+)+`
- Examples: `1`, `1.1`, `1.2.1`, `2.3.4.5`

---

## 🏗️ Data Structure & Patterns

### **Input: PDF Document**
```
Raw PDF with:
- Hierarchical section headers (1.0, 1.1, 1.1.1, etc.)
- Continuous text content
- Embedded tables with or without captions
- Mixed page layouts
```

### **Processing: SectionNode Tree**
```python
class SectionNode:
    section_id: str          # "1.2.1"
    title: str               # "Concrete Mix Design"
    content: list[str]       # [text fragments...]
    tables: list[dict]       # [{"table_id": "Table 4.1", "data": [...]}]
    children: list[SectionNode]  # child sections
```

### **Output: Flattened Chunks (parsed_sections.json)**
```json
{
  "id": "1.2.1_text_0",
  "section": "1.2.1",
  "title": "Concrete Mix Design",
  "text": "Content text here...",
  "metadata": {
    "section_id": "1.2.1",
    "parent": "1.2",
    "hierarchy_path": "1 > 1.2 > 1.2.1",
    "depth": 3,
    "chunk_type": "section_content",
    "content_length": 1245,
    "word_count": 187,
    "sentence_count": 12,
    "is_continuation": false,
    "chunk_index": 0,
    "total_chunks": 2
  }
}
```

---

## 📊 How Tables Are Stored

### **1. Table Detection**
```
PDF Page
  ↓
PyMuPDF finds table boundaries (bounding boxes)
  ↓
Extracts raw table data: list[list[str]]
  ↓
Each cell = string (handles multi-line → space-separated)
```

### **2. Caption Association**

**Priority Order:**
```
1. Explicit text caption in previous line
   "TABLE 4.1 Graded Stone Aggregate"
        ↓
   [stored as table_id]

2. Embedded caption (first row of table)
   ["TABLE 4.1 Description", "", ...]
        ↓
   Extract and remove from table

3. Fallback
   "Table (unknown)"
```

### **3. Table Storage Format**

```python
{
  "type": "table",
  "table_id": "Table 4.1 Graded Stone Aggregate",
  "data": [
    ["Column 1", "Column 2", "Column 3"],
    ["Row 1 Val 1", "Row 1 Val 2", "Row 1 Val 3"],
    ["Row 2 Val 1", "Row 2 Val 2", "Row 2 Val 3"],
    ...
  ],
  "y": 450.25,      # vertical position on page
  "page": 5         # page number
}
```

### **4. Rich Table Metadata**

```json
{
  "id": "1.2_table_0",
  "section": "1.2",
  "title": "Mix Specifications",
  "text": "Mix Specifications\nTable 4.1: Columns: Column 1, Column 2, Column 3 (+2 more). Rows: ~15\n...[preview rows]...",
  "table_data": [["raw", "table", "data"]],
  "metadata": {
    "chunk_type": "table",
    "table_id": "Table 4.1 Mix Specifications",
    "table_index": 0,
    "table_rows": 15,
    "table_cols": 5,
    "table_description": "Table 4.1: Columns: Column 1, Column 2, Column 3 (+2 more). Rows: ~15",
    "hierarchy_path": "1 > 1.2",
    "depth": 2,
    ...
  }
}
```

### **5. Cross-Page Table Merging**

```
Page 1:          Page 2:
┌──────────┐    ┌──────────┐
│ Table... │    │ Unknown  │
│ Row 1    │    │ Table    │
│ Row 2    │    │ Row 3    │
│ Row 3    │    │ Row 4    │
└──────────┘    └──────────┘
      ↓               ↓
[Heuristics]
- Same column count?
- Second has "unknown" ID?
- Consecutive pages?
      ↓
    [MERGE]
      ↓
Final: Table with Rows 1-4
```

---

## 🔄 Semantic Chunking Strategy

### **Why Chunk?**
- Long sections can't fit in embeddings window (~512 tokens)
- Need to split but preserve context

### **How It Works**

```python
# Split on sentence boundaries (not arbitrary char boundaries)
re.split(r'(?<=[.!?])\s+', content)

# Then add OVERLAP for context continuity
```

### **Parameters**
- **max_chunk_size:** 800 characters per chunk
- **overlap_size:** 150 characters of previous chunk repeated

**Example:**
```
Original: [Sentence 1. Sentence 2. Sentence 3. Sentence 4.]

Chunk 0: "Sentence 1. Sentence 2."
           └─────────────────┘

Chunk 1: "Sentence 2. Sentence 3. Sentence 4."
          └─────────────────┘ (overlap added)
```

---

## 📈 Rich Metadata (What's Tracked)

### **For Each Chunk:**

| Field | Purpose | Example |
|-------|---------|---------|
| `id` | Unique identifier | `"1.2.1_text_0"` |
| `section_id` | Which section | `"1.2.1"` |
| `hierarchy_path` | Full context path | `"1 > 1.2 > 1.2.1"` |
| `depth` | Nesting level | `3` |
| `chunk_type` | "section_content" or "table" | `"section_content"` |
| `content_length` | Characters | `1245` |
| `word_count` | Words | `187` |
| `sentence_count` | Sentences | `12` |
| `is_continuation` | Split from long content? | `false` |
| `chunk_index` | Position in split | `0` |

---

## 🎯 Key Quality Checks (Filtering Noise)

```python
def is_valid_section(text):
    # ❌ Filter out pure measurements: "50 mm", "2.5 kg", "mm²"
    # ❌ Filter out >40% digits: page numbers, indices
    # ❌ Filter out non-alphabetic: pure symbols/numbers
    # ❌ Filter out short text: <5 characters
    # ❌ Filter out metadata: "page", "chapter", "index", etc.
    # ✅ Keep meaningful content
```

---

## 📊 Output Statistics

When parser.py runs, it generates:

```
Total Chunks: 847
├─ Section Chunks: 621
├─ Table Chunks: 226
└─ Total Content Size: ~420,000 characters

Depth Distribution:
├─ Level 0: 12 chunks
├─ Level 1: 145 chunks
├─ Level 2: 287 chunks
├─ Level 3: 265 chunks
└─ Level 4+: 138 chunks

Output File: parsed_sections.json (enriched with full metadata)
```

---

## 🔗 How It Feeds Into RAG Pipeline

```
parser.py Output (parsed_sections.json)
     ↓
  ┌──────────────────────┐
  │ Each chunk becomes:  │
  ├──────────────────────┤
  │ 1. Embedded (vector) │ ← Stored in ChromaDB
  │ 2. Metadata indexed  │ ← For retrieval
  │ 3. Full text stored  │ ← For LLM context
  └──────────────────────┘
     ↓
retrieve_and_generate.py
     ↓
RAG system finds relevant chunks + generates answers
```

---

## 💡 Key Innovations

1. **Semantic Overlap** - Chunks overlap to preserve context across splits
2. **Hierarchy Preservation** - Full path (1 > 1.2 > 1.2.1) included in metadata
3. **Table Merging** - Handles tables split across page boundaries
4. **Reading Order** - Maintains top-to-bottom order using coordinates
5. **Rich Metadata** - Every chunk tagged with depth, type, stats, hierarchy
6. **Noise Filtering** - Multi-layer validation removes page numbers, units, etc.

---

## 🚀 Video Script Talking Points

**Opening:**
> "This project uses a three-phase extraction pipeline to convert a PDF specification document into a knowledge base for AI retrieval."

**Phase 1 - Extraction:**
> "Using PyMuPDF, we scan every page, detect tables by their bounding boxes, and extract text while preserving reading order using Y-coordinates."

**Phase 2 - Hierarchization:**
> "We build a tree structure from numbered section headers (1.0, 1.1, 1.2.1) and link tables to their parent sections for context."

**Phase 3 - Semantic Chunking:**
> "Long sections are intelligently split on sentence boundaries with 150-character overlap to ensure context isn't lost during embedding."

**Tables:**
> "Tables are detected, captions are linked automatically, and cross-page tables are merged using smart heuristics."

**Output:**
> "Each chunk gets rich metadata: hierarchy path, depth, word count, and content statistics. This information helps the RAG system retrieve more relevant context."

---

## 📁 File Output: parsed_sections.json

```json
[
  {
    "id": "1_text",
    "section": "1",
    "title": "Materials",
    "text": "Materials\n...",
    "metadata": { /* all tracking info */ }
  },
  {
    "id": "1.1_text",
    "section": "1.1",
    "title": "Cement",
    "text": "Cement\n...",
    "metadata": { /* all tracking info */ }
  },
  {
    "id": "1.2_table_0",
    "section": "1.2",
    "title": "Specifications",
    "text": "Specifications\nTable 4.1: ...",
    "table_data": [[...]], 
    "metadata": { /* all tracking info */ }
  }
]
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.8+ |
| **Core Library** | PyMuPDF (fitz) |
| **Input** | PDF file |
| **Output** | JSON with chunks + metadata |
| **Strategy** | Hierarchical + semantic chunking |
| **ML Models** | None (pure rule-based) |
| **Key Pattern** | Section IDs with dot notation (1.2.3) |
| **Chunk Size** | 800 chars max, 150 char overlap |
| **Table Detection** | Bounding box geometry + regex captions |
| **Noise Filter** | 6-layer validation system |
| **Metadata Tracked** | 15+ fields per chunk |
