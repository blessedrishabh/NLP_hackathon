# RAG Method Statement Generator for RCC Works

A **Retrieval-Augmented Generation (RAG) pipeline** that transforms CPWD construction specification PDFs into semantically-aware, AI-generated Method Statements for Reinforced Cement Concrete (RCC) works.

## 🎯 What This Project Does

This end-to-end NLP pipeline consists of **4 processing stages**:

```
PDF Specification
     ↓
[Stage 1] Parse    → Extract text & tables with semantic chunking (v3 metadata)
     ↓
[Stage 2] Embed    → Generate embeddings, enrich metadata, store in ChromaDB
     ↓
[Stage 3] Retrieve → Multi-query semantic search + cross-encoder reranking
     ↓
[Stage 4] Generate → Groq LLM generates Method Statement sections + accuracy metrics
     ↓
DOCX Output (+ debug JSON + metrics)
```

### Key Features

- **Intelligent PDF Parsing** — Hierarchical chunking, table detection, cross-page table merging
- **Semantic Embeddings** — SentenceTransformer (`all-MiniLM-L12-v2`) for context-aware search
- **Multi-Pass Retrieval** — Prose + tables separately, then cross-encoder reranking
- **LLM Generation** — Groq's Llama-3.3-70B for faithful, cited output
- **Accuracy Metrics** — Jaccard + BERT Score evaluation
- **Professional DOCX** — Styled Word document with 10 Method Statement sections

---

## 📋 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Groq API Key** (free tier at https://console.groq.com)
- **PDF file** named `Prescriptive Specifications_CPWD.pdf` (or rename in config)

### 2. Install Dependencies

```bash
pip install pymupdf chromadb sentence-transformers groq python-docx bert-score python-dotenv
```

**Optional:** If running in Google Colab, the notebook cell handles installation.

### 3. Set Groq API Key

**Option A: Environment Variable**
```bash
export GROQ_API_KEY="your_api_key_here"
```

**Option B: `.env` file**
```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
```

**Option C: Direct in code** (see Configuration section below)

### 4. Place Your PDF

Copy your CPWD specification PDF to the project directory:
```
d:\NLP_Hackathon\project\Prescriptive Specifications_CPWD.pdf
```

### 5. Run the Pipeline

**Option A: Google Colab (Recommended)**
- Open `RAG_Method_Statement_Pipeline.ipynb` in Google Colab
- Upload your PDF when prompted
- Set `GROQ_API_KEY` in the Configuration cell
- Run all cells (`Runtime → Run all`)

**Option B: Jupyter Notebook (Local)**
```bash
jupyter notebook RAG_Method_Statement_Pipeline.ipynb
```

**Option C: Command Line (Individual Scripts)**

Run stages sequentially:

```bash
# Stage 1: Parse PDF → parsed_sections.json
python parser.py Prescriptive_Specifications_CPWD.pdf

# Stage 2: Embed & Store → chroma_db/
python embed_and_store.py parsed_sections.json chroma_db

# Stage 3: Retrieve & Generate → output/generated_sections_debug.json
python retrieve_and_generate.py chroma_db parsed_sections.json --api_key "your_key"

# Stage 4: Generate DOCX (Python-based, replaces generate_docx.js)
# Use the generate_docx.py script or run via notebook
```

---

## 📂 Project Structure

```
d:\NLP_Hackathon\project\
├── README.md                                 # This file
├── RAG_Method_Statement_Pipeline.ipynb       # All-in-one Colab notebook
├── parser.py                                 # Stage 1: PDF parsing
├── embed_and_store.py                        # Stage 2: Embedding & storage
├── retrieve_and_generate.py                  # Stage 3: Retrieval & generation
├── generate_docx.py                          # Stage 4: DOCX generation
├── requirements.txt                          # Python dependencies
├── PARSER_STRATEGY_GUIDE.md                  # Detailed parser documentation
├── EMBED_AND_STORE_STRATEGY_GUIDE.md         # Detailed embed_and_store docs
├── RETRIEVE_AND_GENERATE_STRATEGY_GUIDE.md   # Detailed retrieval & generation docs
├── parsed_sections.json                      # Stage 1 output (v3 format chunks)
├── chroma_db/                                # Stage 2 output (vector database)
│   ├── chroma.sqlite3
│   └── d7030043-7a48-4c09-a763-c97b144a91a1/
├── output/
│   ├── Method_Statement_RCC.docx             # Final output (Word document)
│   ├── generated_sections_debug.json         # LLM-generated content per section
│   ├── retrieved_chunks_debug.json           # Retrieved chunks per section
│   └── accuracy_metrics.json                 # Jaccard + BERT scores
└── package.json                              # Node.js config (legacy generate_docx.js)
```

---

## 🔄 Pipeline Stages Explained

### Stage 1: PDF Parsing (`parser.py`)

**Input:** PDF file
**Output:** `parsed_sections.json` (847 chunks with rich metadata)

**What it does:**
- Extracts text & tables from PDF using PyMuPDF
- Detects hierarchical section structure (1.0, 1.1, 1.2.1, etc.)
- Builds parent-child section tree
- Applies semantic chunking (splits long sections on sentence boundaries with 150-char overlap)
- Merges cross-page tables
- Adds rich metadata: hierarchy path, depth, content stats

**Key innovation:** Chunks preserve hierarchical context for better retrieval.

---

### Stage 2: Embedding & Storage (`embed_and_store.py`)

**Input:** `parsed_sections.json`
**Output:** ChromaDB vector store + enriched chunks

**What it does:**
- Detects v2 or v3 chunk format (auto-converts if needed)
- Classifies chunks into 8 semantic categories (procedure, equipment, quality, etc.)
- Enriches metadata: importance score, content density, chunk position
- Generates 384-dimensional embeddings (SentenceTransformer)
- Deduplicates IDs (handles semantic chunking collisions)
- Upserts into ChromaDB for fast similarity search

**Statistics example:**
- 847 total chunks → 621 section + 226 table
- 8 semantic categories tracked
- ~420,000 total characters

---

### Stage 3: Retrieval & Generation (`retrieve_and_generate.py`)

**Input:** ChromaDB + queries for 10 MS sections
**Output:** Generated content + retrieved chunks + accuracy metrics

**What it does (per section):**
1. **Multi-Query Retrieval** — Runs 2-8 targeted queries for each section
2. **Prose + Table Passes** — Separates general content from dense tables
3. **Cross-Encoder Reranking** — Scores chunks contextually (more accurate than cosine distance)
4. **Importance Boosting** — Elevates top-level sections & table chunks
5. **Top-16 Selection** — Returns highest-scoring 16 unique chunks
6. **LLM Generation** — Calls Groq (Llama-3.3-70B) with context + system prompt
7. **Accuracy Evaluation** — Computes Jaccard token overlap + BERT F1 score

**Output files:**
- `generated_sections_debug.json` — All 10 sections with LLM output
- `retrieved_chunks_debug.json` — Retrieved chunks per section
- `accuracy_metrics.json` — Jaccard + BERT scores

---

### Stage 4: DOCX Generation

**Input:** Generated sections (JSON)
**Output:** `Method_Statement_RCC.docx` (styled Word document)

**What it does:**
- Creates branded cover page (team name, date, members)
- Styles 10 Method Statement sections with navy headings + rules
- Parses inline **bold** markers and lists (bullets/numbered)
- Adds "Document Conventions" notes page
- Outputs professional Word document

---

## ⚙️ Configuration

Edit these in the notebook's **Configuration cell** or script headers:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `GROQ_API_KEY` | *required* | Groq API key for LLM access |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L12-v2` | Sentence-transformer model |
| `N_PER_QUERY` | 8 | Chunks retrieved per query |
| `MAX_CONTEXT_CHARS` | 16000 | Max context window for LLM |
| `RETRIEVAL_POOL_CAP` | 16 | Top-K chunks per section |
| `TEAM_NAME` | `"Musketeers"` | Displayed on cover page |
| `TEAM_MEMBERS` | *list* | Team member names |
| `PDF_PATH` | `"Prescriptive Specifications_CPWD.pdf"` | Input PDF filename |

---

## 📊 Expected Outputs

### 1. `parsed_sections.json` (Stage 1)
```json
{
  "id": "1.2.1_text_0",
  "section": "1.2.1",
  "title": "Concrete Mix Design",
  "text": "Mix Design methodology...",
  "metadata": {
    "section_id": "1.2.1",
    "hierarchy_path": "1 > 1.2 > 1.2.1",
    "depth": 3,
    "chunk_type": "section_content",
    "word_count": 187,
    ...
  }
}
```

### 2. ChromaDB (Stage 2)
```
chroma_db/
├── chroma.sqlite3          (embeddings store)
└── d7030043-7a48.../       (HNSW index for fast search)
```

### 3. `generated_sections_debug.json` (Stage 3)
```json
{
  "purpose": "The Method Statement defines...",
  "scope": "This covers all RCC works...",
  "procedure": "Step-by-step concreting process...",
  ...
}
```

### 4. `accuracy_metrics.json` (Stage 3)
```json
{
  "purpose": {
    "jaccard": 0.0891,
    "bert_f1": 0.6234,
    "n_chunks": 12
  },
  "_summary": {
    "avg_jaccard": 0.1023,
    "avg_bert_f1": 0.6512,
    "llm_model": "llama-3.3-70b-versatile"
  }
}
```

### 5. `Method_Statement_RCC.docx` (Stage 4)
Professional Word document with:
- Cover page (team info, date)
- 10 Method Statement sections
- Styled headings, lists, tables
- Notes page (document conventions, parameters)
- Page numbers & header/footer

---

## 🚀 Running on Google Colab (Recommended)

1. **Open the notebook**
   - Go to `RAG_Method_Statement_Pipeline.ipynb`
   - Open in Google Colab

2. **Install & Import** (Cell 1-2)
   - Automatically installs all dependencies

3. **Configuration** (Cell 3)
   - Paste your Groq API key
   - Set team name & members
   - Upload your PDF using the widget

4. **Run All Cells** (Cell 1-8)
   - `Runtime → Run all` or `Ctrl+F9`
   - Total runtime: ~10-20 minutes

5. **Download Outputs**
   - Files auto-download at the end (Cell 8)
   - Or access from `/output/` directory

---

## 📈 Performance & Accuracy

### Runtime
- **Stage 1 (Parse):** ~1-2 minutes (PDF extraction)
- **Stage 2 (Embed):** ~2-3 minutes (844 embeddings)
- **Stage 3 (Generate):** ~20-40 seconds per section (Groq API calls)
- **Stage 4 (DOCX):** ~5 seconds
- **Total:** ~15-30 minutes

### Accuracy Metrics (Typical)
- **Jaccard Token Overlap:** 0.08-0.15 (8-15% direct token match)
- **BERT F1 Score:** 0.55-0.75 (55-75% semantic similarity)

*Note: Metrics measure faithfulness to retrieved specification text, not ground truth.*

---

## 🛠️ Troubleshooting

### "Groq API Key not found"
- Set `GROQ_API_KEY` environment variable or `.env` file
- Get free key at https://console.groq.com

### "PDF not found"
- Ensure `Prescriptive Specifications_CPWD.pdf` is in the project directory
- Or update `PDF_PATH` in the configuration

### "ChromaDB collection already exists"
- Delete `chroma_db/` folder to force a fresh re-embedding
- Or modify the code to `collection = client.get_collection(...)` instead of `get_or_create_collection`

### "bert-score import error"
- Optional package; ignore warning or install: `pip install bert-score`
- Metrics will skip BERT scores if unavailable

### "Out of memory (GPU)"
- No GPU required; runs on CPU
- Reduce `N_PER_QUERY` or `MAX_CONTEXT_CHARS` if memory is constrained

---

## 📚 Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Extraction** | PyMuPDF (fitz) | Text & table parsing from PDF |
| **Embeddings** | SentenceTransformer | Semantic vector encoding (384-dim) |
| **Vector DB** | ChromaDB | Persistent vector store + HNSW index |
| **LLM** | Groq (Llama-3.3-70B) | Fast inference for Method Statement generation |
| **Reranking** | CrossEncoder (ms-marco) | Contextual ranking of retrieved chunks |
| **Evaluation** | BERT Score | Semantic similarity measurement |
| **DOCX Generation** | python-docx | Professional Word document styling |
| **Language** | Python 3.8+ | Orchestration of entire pipeline |

---

## 📖 Detailed Documentation

For in-depth explanations of each stage, see:

- [**PARSER_STRATEGY_GUIDE.md**](PARSER_STRATEGY_GUIDE.md) — PDF extraction, hierarchical chunking, table detection
- [**EMBED_AND_STORE_STRATEGY_GUIDE.md**](EMBED_AND_STORE_STRATEGY_GUIDE.md) — Embedding generation, ChromaDB storage, metadata enrichment
- [**RETRIEVE_AND_GENERATE_STRATEGY_GUIDE.md**](RETRIEVE_AND_GENERATE_STRATEGY_GUIDE.md) — Multi-pass retrieval, LLM generation, accuracy metrics

---

## 👥 Team

**Project:** RAG Method Statement Generator for CPWD RCC Works  
**Team:** Musketeers  
**Members:**
- Rishabh Sharma (Team Leader)
- Aman Likhitkar
- Rishabh Bharadwaj
- Deepak Pachauri

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 🔗 References

- [ChromaDB Documentation](https://docs.trychroma.com)
- [SentenceTransformer Models](https://www.sbert.net/docs/pretrained_models.html)
- [Groq API Docs](https://console.groq.com/docs)
- [python-docx Guide](https://python-docx.readthedocs.io)
- [IS 456:2000 — Code of Practice for Plain and Reinforced Concrete](https://en.wikipedia.org/wiki/IS_456)

---

**Last Updated:** May 2026  
**Status:** Production Ready ✅
