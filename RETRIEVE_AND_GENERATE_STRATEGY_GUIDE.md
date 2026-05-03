# Retrieve & Generate Pipeline - Video Explanation Guide

## 🎯 Project Strategy Overview

**retrieve_and_generate.py** is the **final stage** of your RAG (Retrieval-Augmented Generation) pipeline. It's the **"question-answering engine"** that:

1. **Retrieves** relevant chunks from ChromaDB based on semantic queries
2. **Reranks** results using cross-encoder intelligence
3. **Generates** a structured Method Statement document using an LLM (Groq)
4. **Evaluates** accuracy using Jaccard overlap and BERT scores

```
Query about "concrete mixing"
     ↓
[Stage 4] RETRIEVAL
  Embed query → Search ChromaDB → Get top chunks
     ↓
[Stage 5] GENERATION
  Pass chunks to LLM → Generate section text
     ↓
[Stage 6] EVALUATION
  Compare generated text vs. retrieved chunks → Accuracy metrics
```

---

## 📚 Core Strategy: "Retrieve Smart, Generate Faithfully, Measure Accuracy"

### **Three-Stage RAG Pipeline**

```
Stage 4: Multi-Pass Semantic Retrieval    Stage 5: Context-Aware Generation    Stage 6: Accuracy Evaluation
──────────────────────────────────────    ────────────────────────────────────    ─────────────────────────
1. Encode query using SentenceTransformer  1. Build context from retrieved chunks  1. Jaccard token overlap
2. Query ChromaDB (prose + tables)         2. Call Groq LLM with system prompt     2. BERT semantic similarity
3. Cross-encoder reranking                 3. Enforce direct citation format       3. Generate accuracy report
4. Importance-score boosting               4. Require ≥40% direct phrases
5. Return top 16 unique chunks
```

---

## 🔧 Modules & Libraries Used

### **1. ChromaDB (Vector Database)**
```python
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("sections")
collection.query(query_embeddings=[...], n_results=8)
```

**What it does:**
- Stores embeddings + metadata + documents from embed_and_store.py
- Performs fast similarity search using HNSW index
- Filters by metadata (e.g., chunk_type, depth)
- Returns matching chunks with distances

### **2. SentenceTransformer (Query Encoding)**
```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Encode queries to same embedding space as stored chunks
model = SentenceTransformer("all-MiniLM-L12-v2")
embedding = model.encode([query])

# Rerank results using contextual understanding
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, doc) for doc in docs])
```

**What it does:**
- **SentenceTransformer:** Converts query text → 384-dim vector (matches embed_and_store)
- **CrossEncoder:** Scores query-document pairs for semantic relevance (0→1)
- More accurate than simple cosine distance

### **3. Groq (LLM API)**
```python
from groq import Groq

client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
)
```

**What it does:**
- Fast LLM API (faster than OpenAI, runs on Groq's inference hardware)
- Model: **Llama-3.3-70B** (open-source, free tier available)
- Generates technical Method Statement sections
- Supports system prompts for instruction following

### **4. BERT Score (Accuracy Evaluation)**
```python
from bert_score import score as bert_score_fn

P, R, F1 = bert_score_fn(
    cands=[generated_text],
    refs=[retrieved_context],
    lang="en"
)
# Returns Precision, Recall, F1 scores (0→1)
```

**What it does:**
- Measures semantic similarity using contextual embeddings (RoBERTa)
- Better than exact token match (Jaccard)
- Gives Precision, Recall, F1 scores
- Gracefully skipped if not installed

### **5. Standard Python Libraries**
- `re` - regex for text processing
- `json` - serialize results
- `argparse` - CLI arguments
- `dotenv` - load GROQ_API_KEY from `.env` file

---

## 🏗️ Data Flow & Structures

### **Stage 1: Input - Method Statement Sections**

Each section has queries and generation instructions:

```python
MS_SECTIONS = [
    {
        "key": "purpose",
        "heading": "1. Purpose of the Method Statement",
        "queries": [
            "purpose scope objective of reinforced cement concrete RCC work",
            "general requirements for RCC construction specification",
            "intent application method statement RCC structural concrete",
        ],
        "word_limit": 120,
        "instruction": "Write a concise purpose statement (2–3 sentences)...",
    },
    {
        "key": "procedure",
        "heading": "5. Procedure for Concreting",
        "queries": [
            "procedure batching mixing transporting placing compacting curing concrete",
            "concrete mix design proportioning water cement ratio aggregate",
            "formwork shuttering reinforcement bar bending placing cover",
            # ... 7 more specialized queries
        ],
        "word_limit": 600,
        "instruction": "Write step-by-step: (a) Formwork, (b) Reinforcement, ...",
    },
    # ... 8 more sections total
]
```

**Structure:**
- `key` - internal identifier (used in output)
- `queries` - 2-8 semantic search queries (targeted per-section)
- `word_limit` - target length for generated text
- `instruction` - detailed prompt to LLM for this section

### **Stage 2: Retrieval - Retrieved Chunks**

```json
[
  {
    "id": "1.2.1_text_0",
    "section": "1.2.1",
    "title": "Concrete Mix Design",
    "chunk_type": "section_content",
    "hierarchy_path": "1 > 1.2 > 1.2.1",
    "depth": 3,
    "label": "procedure",
    "importance": 0.9,
    "has_tables": true,
    "is_continuation": false,
    "text": "Concrete mix design shall use 1:2:4 ratio...",
    "distance": 0.1234,
    "final_score": 0.8567
  },
  {
    "id": "4.2_table_1",
    "section": "4.2",
    "title": "Mix Specifications",
    "chunk_type": "table",
    "table_id": "Table 5.1 Mix Ratios",
    "text": "Cement | Sand | Aggregate\n25 | 50 | 75\n...",
    "hierarchy_path": "4 > 4.2",
    "distance": 0.2156,
    "final_score": 0.7823
  }
]
```

**Key fields:**
- `hierarchy_path` - full section path (e.g., "5 > 5.1 > 5.1.2")
- `chunk_type` - "section_content" or "table" (from parser v3)
- `final_score` - combined score = cross_encoder_score × importance_boost
- `distance` - ChromaDB similarity (lower = closer)

### **Stage 3: Context - Formatted for LLM**

```
=== SPECIFICATION TEXT ===
[1] PATH: 1 > 1.2 > 1.2.1 — Concrete Mix Design
Concrete mix design shall use 1:2:4 ratio. Portland cement OPC 43 grade.
Water-cement ratio 0.45-0.55. Slump 100-150 mm.

[2] PATH: 2 > 2.1 — Mixing Requirements
Concrete shall be mixed for a minimum of 2 minutes. Use a pan mixer.

=== SPECIFICATION TABLES ===
[3] PATH: 4 > 4.2 — Mix Specifications
**Table 5.1 Mix Ratios**
| Cement | Sand | Aggregate | Water |
|--------|------|-----------|-------|
| 25 | 50 | 75 | 12 |
```

**Format:**
- Chunk number: `[1]`, `[2]`, `[3]`
- Full hierarchy path for precise citations
- Prose blocks → plain text
- Table chunks → markdown format
- Max 16,000 characters total

### **Stage 4: Output - Generated Content**

```json
{
  "purpose": "The Method Statement defines the systematic approach for executing reinforced cement concrete (RCC) works as per the CPWD specification. It ensures structural integrity, quality control, and adherence to Indian Standards (IS 456:2000).",
  
  "scope": "This Method Statement covers all RCC works including foundations, columns, beams, and slabs of M25 and M30 grades, from formwork setup through final curing.",
  
  "procedure": "Concreting procedure:
(a) Formwork: Design shall be per IS 14959. Props at 1.0 m × 1.5 m spacing...
(b) Reinforcement: Grade Fe 500, cover 40 mm for concrete exposed...
(c) Mix Design: M25 = 1:1.75:3.5 (1:1.75:3.5 + 0.5 water per IS 10262)...
(d) Mixing: Batch size 0.5 m³ using drum mixer...
(e) Placing: Slump 75-100 mm, drop height ≤1.5 m...
(f) Compaction: External vibration for 15-20 seconds per placement zone...
(g) Finishing: Trowel smooth, remove air voids...
(h) Curing: Water curing for 14 days per IS 456:2000...",
  
  "quality": "Sampling: 1 cube per 50 m³ of concrete. Testing at 7 and 28 days per IS 516. Acceptance: ≥90% of 28-day strength as per IS 456."
}
```

### **Stage 5: Metrics - Accuracy Evaluation**

```json
{
  "purpose": {
    "jaccard": 0.0891,
    "n_chunks": 8,
    "n_table_chunks": 1,
    "bert_f1": 0.5234,
    "bert_p": 0.6123,
    "bert_r": 0.4567
  },
  "scope": {
    "jaccard": 0.1245,
    "n_chunks": 6,
    "n_table_chunks": 0,
    "bert_f1": 0.6012,
    "bert_p": 0.7234,
    "bert_r": 0.5123
  },
  "_summary": {
    "avg_jaccard": 0.1023,
    "avg_bert_f1": 0.5823,
    "embedding_model": "all-MiniLM-L12-v2",
    "llm_model": "llama-3.3-70b-versatile"
  }
}
```

---

## 🔄 Stage 4: Multi-Pass Semantic Retrieval

### **Algorithm: 3-Pass Retrieval with Reranking**

```python
def retrieve_chunks(collection, queries: list[str], n_per_query: int = 8):
    """
    Multi-pass retrieval:
    1. Unfiltered search (all chunk types)
    2. Table-specific search (ensure tables aren't crowded out)
    3. Cross-encoder reranking
    4. Importance-score boosting
    """
```

### **Pass 1: Unfiltered Semantic Search**

```
For each query in queries:
  1. Encode query using SentenceTransformer
     "procedure batching mixing concrete" 
         ↓ (384-dim vector)
     [0.123, -0.456, 0.789, ...]
  
  2. Query ChromaDB
     collection.query(
       query_embeddings=[embedding],
       n_results=8
     )
  
  3. Filter by DISTANCE_THRESHOLD (1.4)
     Keep only chunks with distance ≤ 1.4
  
  4. Avoid duplicates
     Track seen chunk IDs, skip if seen before
```

**Distance metric:**
- ChromaDB uses cosine distance (0→2, lower = more similar)
- 0.1 = very similar
- 1.0 = orthogonal
- 1.4 = threshold (generous window)

### **Pass 2: Table-Specific Search**

```
For each query in queries:
  1. Same as Pass 1, but with filter:
     where: {"chunk_type": "table"}
  
  2. Retrieve additional 4 table chunks
     (ensures tables not crowded out by prose)
  
  3. Merge with unfiltered results (avoid dups)
```

**Why?** Tables are dense information sources but may lose to prose in similarity scoring. This ensures we get both high-similarity prose AND relevant tables.

### **Pass 3: Cross-Encoder Reranking**

```python
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Combine all query terms into one query
combined_query = "procedure batching mixing transporting placing..."

# Score all retrieved chunks against combined query
pairs = [(combined_query, chunk_text) for chunk in raw_chunks]
ce_scores = reranker.predict(pairs)  # → scores in range [0, 1]
```

**Why?**
- Simple cosine distance only considers query-document similarity
- Cross-encoder considers *full query context* + *full document context*
- More accurate ranking

### **Pass 4: Importance Boosting**

```python
def _importance_boost(metadata) -> float:
    """
    Boost score based on metadata from parser v3:
    - importance_score (1.0-1.2 baseline)
    - depth (shallower = more definitive)
    - has_tables (tables carry quantitative data)
    - is_continuation (penalize slightly, but not too much)
    """
    
    score = metadata["importance_score"]  # 1.0 or 1.2
    
    # Depth bonus (top-level sections are foundational)
    if depth == 0:
        score += 0.15  # Top-level, e.g., "1"
    elif depth == 1:
        score += 0.07  # One level deep, e.g., "1.1"
    # depth ≥ 2: no bonus
    
    # Table bonus (quantitative > qualitative)
    if has_tables:
        score += 0.05
    
    # Continuation penalty (but small, 0.05)
    # Continuation chunks can still carry important context
    if is_continuation:
        score -= 0.05
    
    return score
```

**Final scoring:**
```
final_score = cross_encoder_score × importance_boost

Example:
  cross_encoder_score = 0.85
  importance_boost = 1.10  (depth 0 + has_tables)
  final_score = 0.85 × 1.10 = 0.935
```

### **Pass 5: Deduplication & Top-K**

```
1. Sort by final_score (descending)
2. Keep top 16 unique chunks
3. Return with all metadata for context building
```

---

## 🎯 Stage 5: LLM-Powered Generation

### **System Prompt (Instructions to LLM)**

```
You are a technical writer creating a Method Statement for RCC works.

RULES:
1. Use ONLY information from specification excerpts provided.
2. If not enough info: "Not explicitly stated in the specification."
3. Do NOT hallucinate values or requirements.
4. CITATION FORMAT — CRITICAL:
   Use full hierarchy path from chunk header: "§5 > 5.1 > 5.1.2"
   NOT bare numbers like "§5"
5. Chunk headers show "PATH: X > X.Y > X.Y.Z" — use exact path.
6. Quote key terms DIRECTLY from specification.
   Preserve exact wording of requirements.
```

### **User Prompt (Per-Section Instruction)**

```
Example for "Procedure" section:

--- SPECIFICATION EXCERPTS ---
[1] PATH: 1 > 1.2 > 1.2.1 — Concrete Mix Design
Concrete mix design shall use 1:2:4 ratio...

[2] PATH: 5 > 5.1 — Formwork Design
Formwork shall be per IS 14959...

=== SPECIFICATION TABLES ===
[3] PATH: 4 > 4.2 — Mix Specifications
**Table 5.1**
...

--- TASK ---
Write a step-by-step concreting procedure with numbered sub-steps:
(a) Formwork, (b) Reinforcement, (c) Mix Design / Batching,
(d) Mixing, (e) Transportation & Placing, (f) Compaction,
(g) Finishing, (h) Curing.

Include specific values (grades, w/c ratios, slump, curing period)
from the specification where available.

Word limit: approximately 600 words.

CITATION RULE: Every citation MUST use full hierarchy path.

STEP 1 — Before writing, identify 2-3 most relevant sentences/phrases
as verbatim anchors.

STEP 2 — Write the section content now (no heading, just body text).
Use at least 40% direct phrases from retrieved specification text.
Output only the final section body.
```

### **Generation Flow**

```
[System Prompt]
  ↓ (instructs LLM behavior)
[Retrieved Context]
  ↓ (16,000 chars of specs + tables)
[Section Instruction + Word Limit]
  ↓ (specific task for this section)
[Groq LLM (Llama-3.3-70B)]
  ↓ (generates text)
[Generated Section Text]
  ↓ (stored + evaluated)
```

**Parameters:**
- Model: `llama-3.3-70b-versatile` (Groq's open-source 70B model)
- Temperature: 0.2 (low = factual, deterministic)
- Max tokens: 1024 (enough for 150-600 word sections)
- API: Groq (faster inference than OpenAI)

---

## 📊 Stage 6: Accuracy Evaluation

### **Metric 1: Jaccard Token Overlap**

```python
def jaccard(set_a: set, set_b: set) -> float:
    """
    Measure token overlap between generated text and retrieved chunks.
    
    generated_tokens = {"concrete", "batching", "mixing", "procedure", ...}
    retrieved_tokens = {"concrete", "batching", "mixing", "transporting", ...}
    
    overlap = {"concrete", "batching", "mixing"}
    
    jaccard = len(overlap) / len(union)
           = 3 / 10
           = 0.30
    """
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0
```

**What it measures:**
- How much of generated text comes from retrieved chunks
- Higher = more faithful to specification
- Ignores stopwords (the, a, and, or, etc.)

**Interpretation:**
- 0.0 = no overlap (hallucinated content)
- 0.5 = 50% of tokens match
- 1.0 = 100% token match (unlikely)

**Typical range:** 0.08 → 0.15 (8-15% token overlap)

### **Metric 2: BERT Score (F1)**

```python
P, R, F1 = bert_score_fn(
    cands=[generated_text],      # Generated Method Statement section
    refs=[retrieved_context],    # All retrieved chunks combined
    lang="en"
)

# Returns:
# P (Precision)  : How much of generated text is supported by retrieved chunks
# R (Recall)     : How much of retrieved chunks is used in generated text
# F1             : Harmonic mean (P, R)
```

**Interpretation:**
- **Precision (0→1):** If we cite something, is it in the chunks?
- **Recall (0→1):** Did we use all the relevant chunks?
- **F1:** Balance between precision and recall

**Typical range:** 0.55 → 0.75 (55-75% semantic similarity)

**Why different from Jaccard?**
- Jaccard = exact token match
- BERT = semantic match (contextual embeddings)
- BERT score is more lenient (synonyms count as similar)

### **Metrics Summary Report**

```json
{
  "purpose": {
    "jaccard": 0.0891,
    "n_chunks": 8,
    "n_table_chunks": 1,
    "best_chunk": "1.2",
    "bert_f1": 0.5234,
    "bert_p": 0.6123,
    "bert_r": 0.4567
  },
  "_summary": {
    "avg_jaccard": 0.1023,
    "avg_bert_f1": 0.5823,
    "embedding_model": "all-MiniLM-L12-v2",
    "llm_model": "llama-3.3-70b-versatile",
    "n_sections": 10
  }
}
```

**Saved to:** `output/accuracy_metrics.json`

---

## 🔗 Query Design Strategy

### **Targeted Queries Per Section**

Each section has **2-8 semantic queries** designed to retrieve different angles:

**Example: "Procedure" section (7 queries)**

```python
{
  "key": "procedure",
  "queries": [
    # Query 1: General procedure terms
    "procedure batching mixing transporting placing compacting curing concrete",
    
    # Query 2: Mix design specifics
    "concrete mix design proportioning water cement ratio aggregate",
    
    # Query 3: Formwork requirements
    "formwork shuttering reinforcement bar bending placing cover",
    
    # Query 4: Compaction methods
    "compaction vibration consolidation concrete placing procedure",
    
    # Query 5: Curing specifics
    "curing methods period water curing membrane curing concrete",
    
    # Query 6: Construction sequences
    "concrete pour sequence joints construction cold joints",
    
    # Query 7: Additives
    "admixtures plasticizer retarder accelerator concrete mixing",
  ]
}
```

**Why multiple queries?**
- Single query might miss some angles
- "mix design" query gets different chunks than "curing" query
- Combined results give fuller picture
- Each query targets a different aspect

### **Multi-Query Retrieval Flow**

```
Query Set for "Procedure":
├─ Query 1 → Returns 8 chunks (mix-related)
├─ Query 2 → Returns 8 chunks (design-related)
├─ Query 3 → Returns 8 chunks (formwork-related)
├─ Query 4 → Returns 8 chunks (compaction-related)
├─ Query 5 → Returns 8 chunks (curing-related)
├─ Query 6 → Returns 8 chunks (sequence-related)
└─ Query 7 → Returns 8 chunks (admixture-related)

Total: ~56 chunk retrievals
After deduplication: ~20-25 unique chunks
After reranking & top-16 cut: 16 final chunks
     ↓
Passed to LLM as context
```

---

## 🚀 Complete End-to-End Flow

```
Step 1: LOAD CONFIGURATION
  ├─ Load 10 MS_SECTIONS with queries
  ├─ Load ChromaDB (847 chunks)
  ├─ Load embedding model
  └─ Load Groq API key

Step 2: FOR EACH SECTION
  ├─ RETRIEVE
  │  ├─ For each query → embed + ChromaDB search
  │  ├─ Unfiltered pass (prose + tables)
  │  ├─ Table-specific pass (ensure tables)
  │  ├─ Cross-encoder rerank
  │  ├─ Importance boost using v3 metadata
  │  └─ Return top 16 unique chunks
  │
  ├─ BUILD CONTEXT
  │  ├─ Format chunks with hierarchy paths
  │  ├─ Convert tables to markdown
  │  ├─ Combine into 16,000-char context
  │  └─ Mark continuations
  │
  ├─ GENERATE
  │  ├─ Call Groq with system prompt
  │  ├─ Pass context + section instruction
  │  ├─ Set word limit
  │  ├─ Require 40% direct phrases
  │  └─ Return generated text
  │
  └─ EVALUATE
     ├─ Compute Jaccard overlap
     ├─ Compute BERT F1 score
     └─ Store metrics

Step 3: SAVE OUTPUTS
  ├─ generated_sections_debug.json
  │  (10 sections × generated text)
  │
  ├─ retrieved_chunks_debug.json
  │  (10 sections × 16 chunks with metadata)
  │
  └─ accuracy_metrics.json
     (10 sections × metrics + summary)

Step 4: PRINT DIAGNOSTICS
  ├─ Retrieval details (n_chunks, n_tables, hierarchy_paths)
  ├─ Jaccard scores per section (avg 0.08-0.15)
  ├─ BERT scores per section (avg F1 0.55-0.75)
  └─ Summary statistics
```

---

## 💡 Advanced Retrieval Features

### **Hierarchy-Aware Context Building**

```python
def chunks_to_context(chunks: list[dict]) -> str:
    """
    Build context with explicit hierarchy paths so LLM can cite precisely.
    
    Example output:
    
    [1] PATH: 5 > 5.1 > 5.1.2 — Mix Design Procedure
    Concrete shall be mixed using a drum mixer...
    
    [2] PATH: 5 > 5.2 — Curing Requirements
    Curing shall be done for a minimum of 14 days...
    """
```

**Benefits:**
- LLM sees exact section addresses (e.g., "5 > 5.1 > 5.1.2")
- Can cite with full precision: "As per §5 > 5.1 > 5.1.2"
- No ambiguity about which rule applies

### **Table Rendering to Markdown**

```python
def pipe_to_markdown_table(text: str) -> str:
    """
    Convert pipe-delimited table text to markdown for LLM.
    
    Input:  "Column1 | Column2\nValue1 | Value2"
    Output:
    | Column1 | Column2 |
    |---------|---------|
    | Value1  | Value2  |
    """
```

**Why?**
- LLMs understand markdown tables better than raw text
- Clear column structure
- Easy for LLM to extract specific values

### **Continuation Chunk Labeling**

```
[1] PATH: 2 > 2.1 — Mix Procedure
    First part of a long section...

[2] PATH: 2 > 2.1 — Mix Procedure [continued]
    Second part (semantic overlap with previous chunk)...
```

**Why?**
- LLM knows chunks may be fragments
- Can stitch them together coherently
- Avoids repetition or confusion

---

## 🎯 Key Generation Strategies

### **System Prompt: 8 Critical Rules**

```
Rule 1: Use ONLY provided information
Rule 2: Explicitly state if not enough info
Rule 3: Never hallucinate values
Rule 4: Use precise citation format (§X > X.Y > X.Y.Z)
Rule 5: Understand hierarchy paths from chunk headers
Rule 6: Quote key terms DIRECTLY from specification
Rule 7: Preserve exact wording of requirements
Rule 8: Quote key technical terms directly from spec
```

### **User Prompt: 2-Step Strategy**

**Step 1 (Silent):** Identify 2-3 most relevant sentences from chunks
**Step 2 (Output):** Write section using ≥40% direct phrases

This forces:
- Faithfulness (must find phrases in chunks first)
- High token overlap (40% direct from spec)
- Natural readability (writer doesn't just copy-paste)

---

## 📋 Output Files Generated

### **1. generated_sections_debug.json**
```json
{
  "purpose": "The Method Statement defines...",
  "scope": "This covers...",
  "acronyms": "ACRONYM – Definition...",
  "references": "1. IS 456:2000...",
  "procedure": "Step-by-step procedure...",
  ...
}
```

### **2. retrieved_chunks_debug.json**
```json
{
  "purpose": [
    {
      "id": "1.2.1_text_0",
      "hierarchy_path": "1 > 1.2 > 1.2.1",
      "text": "...",
      "distance": 0.1234,
      "final_score": 0.8567
    },
    ...
  ],
  "procedure": [...],
  ...
}
```

### **3. accuracy_metrics.json**
```json
{
  "purpose": {
    "jaccard": 0.0891,
    "bert_f1": 0.5234,
    "n_chunks": 8
  },
  "_summary": {
    "avg_jaccard": 0.1023,
    "avg_bert_f1": 0.5823
  }
}
```

---

## 🎬 Video Script Talking Points

**Opening:**
> "The retrieve_and_generate pipeline completes the RAG system. It takes user queries, finds the most relevant specification chunks, and uses an LLM to generate an accurate, cited Method Statement."

**Stage 4 - Retrieval:**
> "For each section, we run 2-8 targeted queries. Each query is embedded and searched in ChromaDB. We use a two-pass approach: first for prose, then specifically for tables to ensure dense quantitative data isn't missed."

**Multi-Pass Strategy:**
> "Pass 1 retrieves general content. Pass 2 ensures tables are included. A cross-encoder model then reranks results, and we boost scores based on section hierarchy—top-level sections are treated as more foundational."

**Context Building:**
> "The 16 highest-scoring chunks are formatted with full hierarchy paths (e.g., '5 > 5.1 > 5.1.2'), tables are converted to markdown, and everything is combined into a 16,000-character context window."

**Stage 5 - Generation:**
> "We call Groq's Llama-3.3-70B model with a system prompt that enforces rules: use only provided information, cite with full section paths, and quote at least 40% directly from the specification."

**Citation Enforcement:**
> "The LLM is instructed to cite using the full hierarchy path. So instead of vague 'per section 5', it must say 'per §5 > 5.1 > 5.1.2', ensuring traceability back to the exact specification clause."

**Stage 6 - Evaluation:**
> "Two accuracy metrics measure how faithful the generated text is: Jaccard measures token overlap (8-15% typical), while BERT Score measures semantic similarity (55-75% typical). Both saved to accuracy_metrics.json."

**Key Innovation:**
> "The pipeline uniquely leverages parser v3's rich metadata—hierarchy paths, depth, importance scores, and chunk types—to guide intelligent retrieval and context building, resulting in more faithful and traceable Method Statements."

---

## 📊 Performance Characteristics

### **Retrieval Speed**
- Query embedding: ~50ms
- ChromaDB search: ~10-20ms per query
- Cross-encoder reranking: ~100ms for 50 chunks
- **Total per section: ~500-800ms**

### **Generation Speed**
- Groq inference: ~2-5 seconds per section
- **Total 10 sections: ~20-50 seconds**

### **Accuracy Metrics**
- Jaccard: 0.08-0.15 (8-15% token overlap)
- BERT F1: 0.55-0.75 (55-75% semantic similarity)
- Coverage: 16 unique chunks per section

### **Output Sizes**
- generated_sections_debug.json: ~30-50 KB
- retrieved_chunks_debug.json: ~200-300 KB
- accuracy_metrics.json: ~5-10 KB

---

## 🔐 API Key Setup

```bash
# Option 1: Set environment variable
export GROQ_API_KEY="your_api_key_here"

# Option 2: Create .env file
echo "GROQ_API_KEY=your_api_key_here" > .env

# Option 3: Command-line argument
python retrieve_and_generate.py --api_key "your_api_key_here"

# Get free API key at https://console.groq.com
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Input** | ChromaDB embeddings + metadata from embed_and_store |
| **Processing** | Multi-pass semantic retrieval + cross-encoder reranking |
| **LLM** | Groq (Llama-3.3-70B) |
| **Queries Per Section** | 2-8 targeted semantic queries |
| **Chunks Retrieved** | 16 per section (top by reranked score) |
| **Context Size** | ~16,000 characters max |
| **Citation Format** | Full hierarchy path (§X > X.Y > X.Y.Z) |
| **Generation** | System prompt + 8 rules + per-section instruction |
| **Accuracy Metrics** | Jaccard (token overlap) + BERT F1 (semantic) |
| **Output Files** | generated_sections_debug.json, retrieved_chunks_debug.json, accuracy_metrics.json |
| **Total Pipeline Time** | ~30-60 seconds for 10 sections |
| **Key Innovation** | Hierarchy-aware retrieval + precise citation enforcement + dual accuracy metrics |
