"""
retrieve_and_generate.py  (v5 — Jaccard-optimised)
=============================================================================
Stage 4 — Retrieval  : ChromaDB queries per Method-Statement section
Stage 5 — LLM        : Groq (llama-3.3-70b-versatile) writes each section

Changes vs v4 (targeting Jaccard ↑ while keeping BERT F1 stable):
  • n_per_query default 5 → 8  (--n_results CLI flag)
  • chunks_to_context max_chars 12 000 → 16 000
  • Retrieval pool cap 12 → 16 chunks after reranking
  • DISTANCE_THRESHOLD 1.2 → 1.4  (wider retrieval net)
  • Continuation-chunk penalty 0.10 → 0.05  (keep overlap-rich chunks)
  • SYSTEM_PROMPT Rule 8: quote key terms directly from spec
  • call_groq user prompt: snippet-extraction step + "≥40% direct phrases"
  • purpose / scope / references sections: +2 targeted queries each
    (these had the three lowest Jaccard scores: 0.0455 / 0.0611 / 0.0623)

Usage:
    pip install chromadb sentence-transformers groq python-docx bert-score

    python retrieve_and_generate.py \
        --db_dir   chroma_db \
        --out_dir  output \
        --api_key  YOUR_GROQ_API_KEY

    Get free API key at: https://console.groq.com
"""

import os
import re
import json
import argparse
from datetime import datetime

# ── vector store + embeddings ──────────────────────────────────────────────
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── LLM ───────────────────────────────────────────────────────────────────
from groq import Groq

from dotenv import load_dotenv

load_dotenv()

# ── BERT Score (optional — graceful fallback if not installed) ─────────────
try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("⚠  bert-score not installed — BERT Score will be skipped.")
    print("   Install with: pip install bert-score")


# ══════════════════════════════════════════════════════════════════════════
# 0.  EMBEDDING MODEL  (MUST match embed_and_store.py)
# ══════════════════════════════════════════════════════════════════════════
EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"   # ← upgraded from L6 to match v4


# ══════════════════════════════════════════════════════════════════════════
# 1.  SECTION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════
MS_SECTIONS = [
    {
        "key": "purpose",
        "heading": "1. Purpose of the Method Statement",
        "queries": [
            "purpose scope objective of reinforced cement concrete RCC work",
            "general requirements for RCC construction specification",
            # ↓ extra targeted queries to improve Jaccard (was lowest at 0.0455)
            "intent application method statement RCC structural concrete works",
            "description overview reinforced concrete construction project requirement",
        ],
        "word_limit": 120,
        "instruction": (
            "Write a concise purpose statement (2–3 sentences) explaining WHY "
            "this method statement exists, referencing the RCC works covered by "
            "the specification."
        ),
    },
    {
        "key": "scope",
        "heading": "2. Scope of the Method Statement",
        "queries": [
            "scope of work reinforced cement concrete structural elements",
            "applicability RCC specification foundations columns beams slabs",
            # ↓ extra targeted queries to improve Jaccard (was 0.0611)
            "coverage extent types structural members concrete grade specification",
            "inclusions exclusions locations applicable sections concrete works",
        ],
        "word_limit": 150,
        "instruction": (
            "Define WHAT work is covered — structural elements, locations, "
            "grade of concrete — citing the specification sections."
        ),
    },
    {
        "key": "acronyms",
        "heading": "3. Acronyms and Definitions",
        "queries": [
            "abbreviations acronyms definitions terminology concrete specification",
            "IS code BIS OPC PPC w/c ratio slump workability definitions",
        ],
        "word_limit": 200,
        "instruction": (
            "List all acronyms and technical terms found in the retrieved text "
            "as a concise definition list (ACRONYM – full form: succinct meaning).  "
            "CRITICAL formatting rules:\n"
            "  - Each entry must be fully self-contained: state what the acronym "
            "    stands for AND what it means in plain technical language.\n"
            "  - For SCC consistency/flow classes (VS1, VS2, VF1, VF2, SF1–SF3, "
            "    PA0–PA2, PL1–PL2, etc.) write the full class name, then describe "
            "    the property range or test value it represents "
            "    (e.g. 'VS1 – Viscosity Class 1 (SCC): viscosity class requiring a "
            "    V-funnel flow time ≤ 8 s, indicating low-to-moderate viscosity with "
            "    good deformability under own weight').\n"
            "  - Do NOT use dangling clauses such as 'good filling ability even with …' "
            "    without first stating the full form of the acronym and the test parameter.\n"
            "  - Do NOT invent definitions not supported by the retrieved specification text.\n"
            "  - Each definition should be 1–2 sentences max."
        ),
    },
    {
        "key": "references",
        "heading": "4. Reference Documents",
        "queries": [
            "IS code standards references BIS bureau Indian standard concrete",
            "relevant codes specifications standards cited in document",
            # ↓ extra targeted queries to improve Jaccard (was 0.0623)
            "IS 456 IS 383 IS 516 IS 1199 code number referenced specification",
            "standard specification number clause reference concrete reinforcement",
        ],
        "word_limit": 200,
        "instruction": (
            "List every IS code, BIS standard, or other document explicitly "
            "cited in the retrieved chunks.  Format as a numbered list."
        ),
    },
    {
        "key": "procedure",
        "heading": "5. Procedure for Concreting",
        "queries": [
            "procedure batching mixing transporting placing compacting curing concrete",
            "concrete mix design proportioning water cement ratio aggregate",
            "formwork shuttering reinforcement bar bending placing cover",
            "compaction vibration consolidation concrete placing procedure",
            "curing methods period water curing membrane curing concrete",
            "concrete pour sequence joints construction cold joints",
            "admixtures plasticizer retarder accelerator concrete mixing",
        ],
        "word_limit": 600,
        "instruction": (
            "Write a step-by-step concreting procedure with numbered sub-steps: "
            "(a) Formwork, (b) Reinforcement, (c) Mix Design / Batching, "
            "(d) Mixing, (e) Transportation & Placing, (f) Compaction, "
            "(g) Finishing, (h) Curing.  "
            "Include specific values (grades, w/c ratios, slump, curing period) "
            "from the specification where available."
        ),
    },
    {
        "key": "equipment",
        "heading": "6. Equipment Used",
        "queries": [
            "equipment machinery plant concrete mixer batching plant transit mixer pump",
            "vibrator needle poker vibration compaction equipment tools",
            "formwork shuttering props scaffolding equipment",
        ],
        "word_limit": 200,
        "instruction": (
            "List all plant, equipment, and tools mentioned in the specification "
            "for RCC work as a bullet list with a brief note on its use."
        ),
    },
    {
        "key": "personnel",
        "heading": "7. Key People Involved",
        "queries": [
            "personnel responsible engineer supervisor quality control inspection",
            "testing laboratory inspector site engineer foreman concrete",
        ],
        "word_limit": 150,
        "instruction": (
            "List roles/personnel mentioned or implied in the specification "
            "(e.g., site engineer, quality inspector, lab technician) and their "
            "responsibilities."
        ),
    },
    {
        "key": "quality",
        "heading": "8. Quality Control and Testing",
        "queries": [
            "quality control testing cube strength acceptance criteria concrete",
            "sampling frequency cube mould testing compressive strength",
            "inspection checks hold points concrete placement",
        ],
        "word_limit": 250,
        "instruction": (
            "Summarise quality-control requirements: sampling frequency, test "
            "types, acceptance criteria, and any hold/witness points from the "
            "specification."
        ),
    },
    {
        "key": "health_safety",
        "heading": "9. Health, Safety & Environment (HSE) Considerations",
        "queries": [
            "safety health environment precautions hazards concrete construction",
            "PPE protective equipment safety measures workers",
        ],
        "word_limit": 150,
        "instruction": (
            "Summarise HSE measures stated in the specification.  If none are "
            "explicitly stated, note that only and do NOT invent content."
        ),
    },
    {
        "key": "other",
        "heading": "10. Other Relevant Information",
        "queries": [
            "special requirements restrictions notes concrete specification",
            "rejection defective concrete remedial action non-conformance",
            "hot weather cold weather concreting special conditions",
        ],
        "word_limit": 150,
        "instruction": (
            "Include any other specification requirements not covered in the "
            "sections above (e.g. weather conditions, hot/cold weather "
            "concreting, repair of defective work)."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════
# 2.  RETRIEVAL HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_vector_store(db_dir: str):
    """Open the persisted ChromaDB and return the collection."""
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection("sections")
    return collection


def _build_query_text(query: str) -> str:
    """
    Mirror the embed_and_store v4 build_embedding_text() format so that
    query embeddings live in the same semantic space as stored embeddings.

    embed_and_store prepends:
        [Section: {hierarchy_path}]
        Title: {title}
        {text}

    We mimic this by wrapping the plain query in a generic header that
    triggers the same token distribution without forcing a specific section.
    """
    return f"Title: {query}\n{query}"


def _query_collection(collection, embedding: list,
                      n_results: int,
                      chunk_type_filter: str | None = None) -> list[tuple]:
    """
    Single ChromaDB query.
    Returns list of (id, doc, meta, distance).

    chunk_type values in new pipeline:
        "section_content"   ← prose chunks  (was "section" in old pipeline)
        "table"             ← table chunks  (unchanged)
    """
    DISTANCE_THRESHOLD = 1.4   # ↑ from 1.2 — wider net captures more overlapping chunks

    kwargs = dict(
        query_embeddings=embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if chunk_type_filter:
        # new metadata key is "chunk_type" with value "section_content" or "table"
        kwargs["where"] = {"chunk_type": chunk_type_filter}

    try:
        res = collection.query(**kwargs)
    except Exception:
        kwargs["n_results"] = max(1, n_results // 2)
        kwargs.pop("where", None)
        try:
            res = collection.query(**kwargs)
        except Exception:
            return []

    rows = []
    for doc, meta, cid, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["ids"][0],
        res["distances"][0],
    ):
        if dist <= DISTANCE_THRESHOLD:
            rows.append((cid, doc, meta, dist))
    return rows


def _importance_boost(meta: dict) -> float:
    """
    Compute a boost factor from new metadata fields so that hierarchically
    important chunks rise after reranking.

    Factors:
      - importance_score  (1.2 for tables, ≤1.0 for prose, from embed_and_store)
      - depth             (shallower sections tend to be definitional)
      - has_tables        (sections with tables carry quantitative data)
      - is_continuation   (penalise middle-of-long-chunk fragments slightly)
    """
    score = meta.get("importance_score", 1.0)

    depth = meta.get("depth", 2)
    # flatten depth bonus: level-0 +0.15, level-1 +0.07, level-2+ no change
    if depth == 0:
        score += 0.15
    elif depth == 1:
        score += 0.07

    if meta.get("has_tables", False):
        score += 0.05

    if meta.get("is_continuation_chunk", False):
        score -= 0.05   # ↓ reduced from 0.10 — continuation chunks may still carry key terms

    return max(score, 0.1)


def retrieve_chunks(collection, model: SentenceTransformer,
                    queries: list[str], n_per_query: int = 5) -> list[dict]:
    """
    Multi-query retrieval with:
      1. Unfiltered pass (prose + tables)
      2. Table-only pass  (ensures tables not crowded out)
      3. Cross-encoder reranking
      4. Importance-score boosting from v3 metadata
    """
    seen_ids: set[str] = set()
    raw: list[tuple] = []   # (id, doc, meta, distance)

    for query in queries:
        # ── embed using the same prefix format as embed_and_store ──────────
        query_text = _build_query_text(query)
        embedding = model.encode([query_text]).tolist()

        # Pass 1 — all chunk types
        for row in _query_collection(collection, embedding, n_per_query):
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                raw.append(row)

        # Pass 2 — table chunks specifically (corrected filter value)
        for row in _query_collection(
            collection, embedding,
            max(2, n_per_query // 2),
            chunk_type_filter="table",
        ):
            if row[0] not in seen_ids:
                seen_ids.add(row[0])
                raw.append(row)

    if not raw:
        return []

    # ── Cross-encoder reranking ───────────────────────────────────────────
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    combined_query = " ".join(queries)
    pairs = [(combined_query, row[1]) for row in raw]
    ce_scores = reranker.predict(pairs)

    # ── Combine cross-encoder score with importance boost ─────────────────
    boosted = []
    for i, row in enumerate(raw):
        cid, doc, meta, dist = row
        boost = _importance_boost(meta)
        final_score = float(ce_scores[i]) * boost
        boosted.append((cid, doc, meta, dist, final_score))

    boosted.sort(key=lambda r: r[4], reverse=True)
    boosted = boosted[:16]   # ↑ from 12 — larger pool feeds more context to LLM

    # ── Build output dict (expose new metadata fields) ────────────────────
    return [
        {
            "id":              cid,
            "section":         meta.get("section", ""),
            "title":           meta.get("title", ""),
            "chunk_type":      meta.get("chunk_type", "section_content"),
            "table_id":        meta.get("table_id", ""),
            "hierarchy_path":  meta.get("hierarchy_path", ""),
            "depth":           meta.get("depth", 0),
            "label":           meta.get("label", ""),
            "importance":      meta.get("importance_score", 1.0),
            "has_tables":      meta.get("has_tables", False),
            "is_continuation": meta.get("is_continuation_chunk", False),
            "text":            doc,
            "distance":        round(dist, 4),
            "final_score":     round(boosted[i][4], 4),
        }
        for i, (cid, doc, meta, dist, _) in enumerate(boosted)
    ]


# ══════════════════════════════════════════════════════════════════════════
# 3.  CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════

def pipe_to_markdown_table(text: str) -> str:
    """Convert pipe-delimited table text to a markdown table."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return text

    out = []
    data_lines = []
    caption = ""

    if "|" not in lines[0]:
        caption    = lines[0].strip()
        data_lines = lines[1:]
    else:
        data_lines = lines

    if caption:
        out.append(f"**{caption}**")

    if not data_lines:
        return "\n".join(out)

    def _row(cells):
        return "| " + " | ".join(c.strip() for c in cells) + " |"

    header_cells = data_lines[0].split("|")
    out.append(_row(header_cells))
    out.append("| " + " | ".join("---" for _ in header_cells) + " |")

    for row in data_lines[1:]:
        out.append(_row(row.split("|")))

    return "\n".join(out)


def chunks_to_context(chunks: list[dict], max_chars: int = 16000) -> str:
    """
    Serialise retrieved chunks into an LLM context string.

    Improvements over v2:
    - Hierarchy path included in each chunk header
    - Depth and label cues help LLM understand structural context
    - table chunks rendered as markdown
    - Continuation chunks labelled so LLM knows context is split
    - max_chars raised to 16 000 (↑ from 12 000) so more verbatim text
      is visible to the LLM, directly raising Jaccard token overlap
    """
    prose_blocks = []
    table_blocks = []
    total_chars  = 0

    for i, c in enumerate(chunks, 1):
        chunk_type = c.get("chunk_type", "section_content")
        hier       = c.get("hierarchy_path", "")
        cont_note  = " [continued]" if c.get("is_continuation") else ""

        # Lead with hierarchy_path so LLM uses precise path citations (e.g. "5 > 5.1 > 5.1.2")
        # Fall back to flat section number only when path is absent
        path_label = hier if hier else c['section']
        header = (
            f"[{i}] PATH: {path_label} — {c['title']}{cont_note}"
        )

        if chunk_type == "table":
            body  = pipe_to_markdown_table(c["text"])
            block = f"{header}\n{body}\n"
        else:
            block = f"{header}\n{c['text'].strip()}\n"

        if total_chars + len(block) > max_chars:
            break

        if chunk_type == "table":
            table_blocks.append(block)
        else:
            prose_blocks.append(block)

        total_chars += len(block)

    parts = []
    if prose_blocks:
        parts.append("=== SPECIFICATION TEXT ===\n" + "\n".join(prose_blocks))
    if table_blocks:
        parts.append("=== SPECIFICATION TABLES ===\n" + "\n".join(table_blocks))

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# 4.  LLM HELPERS  (Gemini)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a technical writer creating a Method Statement for Reinforced Cement \
Concrete (RCC) works on a construction project.

RULES:
1. Use ONLY information from the specification excerpts provided.
2. If the excerpts do not contain enough information for a section, write:
   "Not explicitly stated in the specification."
3. Do NOT hallucinate values, codes, or requirements.
4. Be concise and professional.
5. CITATION FORMAT — CRITICAL: Every specification reference MUST use the full
   hierarchy path shown in the chunk header (e.g. "per §5 > 5.1 > 5.1.2").
   NEVER cite a bare flat number like "per §3" or "per §10".
   Use the exact "PATH:" value from the chunk header — copy it verbatim.
   Example: "Concrete shall be cured for a minimum of 10 days (per §5 > 5.3 > 5.3.3)."
6. The context contains two blocks:
   - SPECIFICATION TEXT  : prose paragraphs from the spec (hierarchy path shown)
   - SPECIFICATION TABLES: data tables in markdown format
   When referencing table values (sieve sizes, mix ratios, % passing etc.)
   read the column header and row label carefully before quoting a number.
7. Chunk headers show "PATH: X > X.Y > X.Y.Z" — this is the precise section
   address. Always use this full path format when citing a requirement.
8. Quote key technical terms, values, and phrases DIRECTLY from the specification
   excerpts wherever possible (e.g. grades, ratios, period values, code numbers).
   Preserve the exact wording of requirements rather than paraphrasing them.
"""


def call_groq(api_key: str, context: str, section_instruction: str,
              word_limit: int,
              model_name: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq and return the generated text for one MS section."""
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"--- SPECIFICATION EXCERPTS ---\n{context}\n\n"
                f"--- TASK ---\n{section_instruction}\n"
                f"Word limit: approximately {word_limit} words.\n\n"
                f"CITATION RULE: Every citation must use the full hierarchy path "
                f"from the chunk's 'PATH:' label (e.g. '§5 > 5.1 > 5.1.2'). "
                f"Never use a bare flat section number like '§3' or '§10'.\n\n"
                f"STEP 1 — Before writing, silently identify the 2-3 most relevant "
                f"sentences or phrases from the excerpts above that directly answer "
                f"this section.  Keep them in mind as verbatim anchors.\n"
                f"STEP 2 — Write the section content now (no heading, just the body "
                f"text).  Use at least 40% direct phrases and key terms taken "
                f"verbatim from the retrieved specification text.  "
                f"Do NOT add a 'STEP 1' header in your output — output only the "
                f"final section body:"
            )},
        ],
        max_tokens=1024,
        temperature=0.2,   # low temp for factual technical writing
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════
# 5.  ACCURACY METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════

def compute_retrieval_coverage(sections_content: dict[str, str],
                               retrieved_chunks: dict[str, list[dict]]) -> dict:
    """
    Jaccard token-overlap coverage metric (v2, unchanged logic but updated
    to read chunk_type from new metadata field name).
    """
    def tokenise(text: str) -> set[str]:
        STOPWORDS = {"the", "a", "an", "of", "in", "to", "and", "or",
                     "is", "are", "be", "as", "for", "with", "by", "on",
                     "at", "from", "it", "its", "not", "per", "shall"}
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return tokens - STOPWORDS - {t for t in tokens if len(t) <= 1}

    def jaccard(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return round(inter / union, 4) if union else 0.0

    scores = {}
    for key, chunks in retrieved_chunks.items():
        gen_tokens = tokenise(sections_content.get(key, ""))
        if not chunks:
            scores[key] = {"jaccard": None, "n_chunks": 0,
                           "n_table_chunks": 0, "best_chunk": None}
            continue

        jaccard_scores = []
        best_score, best_chunk = -1, None

        for c in chunks:
            chunk_tokens = tokenise(c["text"])
            j = jaccard(gen_tokens, chunk_tokens)
            jaccard_scores.append(j)
            if j > best_score:
                best_score = j
                best_chunk = c["section"]

        mean_j     = round(sum(jaccard_scores) / len(jaccard_scores), 4)
        # new chunk_type value is "table" (unchanged) or "section_content"
        n_table    = sum(1 for c in chunks if c.get("chunk_type") == "table")

        scores[key] = {
            "jaccard":       mean_j,
            "n_chunks":      len(chunks),
            "n_table_chunks": n_table,
            "best_chunk":    best_chunk,
        }

    return scores


def compute_bert_scores(sections_content: dict[str, str],
                        retrieved_chunks: dict[str, list[dict]]) -> dict:
    """
    Compute BERTScore (F1) between each generated section and the
    concatenated text of its retrieved chunks.

    BERTScore uses contextual embeddings (default: roberta-large) to
    measure semantic similarity beyond token overlap, giving a better
    proxy for factual faithfulness.

    Returns: {section_key: {"bert_f1": float, "bert_p": float, "bert_r": float}}
    """
    if not BERT_SCORE_AVAILABLE:
        return {k: {"bert_f1": None, "bert_p": None, "bert_r": None}
                for k in sections_content}

    results = {}
    for key, gen_text in sections_content.items():
        chunks = retrieved_chunks.get(key, [])
        if not chunks or not gen_text.strip():
            results[key] = {"bert_f1": None, "bert_p": None, "bert_r": None}
            continue

        # Reference: all retrieved chunk texts joined
        reference = " ".join(c["text"] for c in chunks if c.get("text"))
        if not reference.strip():
            results[key] = {"bert_f1": None, "bert_p": None, "bert_r": None}
            continue

        try:
            P, R, F1 = bert_score_fn(
                cands=[gen_text],
                refs=[reference],
                lang="en",
                verbose=False,
            )
            results[key] = {
                "bert_f1": round(float(F1[0]), 4),
                "bert_p":  round(float(P[0]), 4),
                "bert_r":  round(float(R[0]), 4),
            }
        except Exception as e:
            results[key] = {"bert_f1": None, "bert_p": None, "bert_r": None,
                            "error": str(e)}

    return results


# ══════════════════════════════════════════════════════════════════════════
# 7.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_pipeline(db_dir: str, api_key: str, out_dir: str,
                 team_name: str = "Your Team Name",
                 n_per_query: int = 8) -> None:   # ↑ from 5 — more chunks → higher Jaccard

    os.makedirs(out_dir, exist_ok=True)

    # ── Load resources ────────────────────────────────────────────────────
    print(f"⏳  Loading embedding model ({EMBEDDING_MODEL_NAME}) …")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("⏳  Connecting to ChromaDB …")
    collection  = load_vector_store(db_dir)
    total_docs  = collection.count()
    print(f"    → {total_docs} chunks in store")

    # ── Per-section retrieval + generation ────────────────────────────────
    sections_content: dict[str, str]        = {}
    all_retrieved:    dict[str, list[dict]] = {}

    for sec in MS_SECTIONS:
        key     = sec["key"]
        heading = sec["heading"]
        print(f"\n📄  Processing: {heading}")

        # Stage 4 — retrieve with new metadata-aware pipeline
        chunks = retrieve_chunks(
            collection, embed_model, sec["queries"], n_per_query
        )
        all_retrieved[key] = chunks

        # Print retrieval diagnostics using new metadata fields
        n_tables = sum(1 for c in chunks if c.get("chunk_type") == "table")
        n_cont   = sum(1 for c in chunks if c.get("is_continuation"))
        print(f"    → {len(chunks)} unique chunks  "
              f"(tables={n_tables}, continuations={n_cont})")

        # Show top chunk hierarchy paths for debugging
        for c in chunks[:3]:
            hier = c.get("hierarchy_path", c.get("section", "?"))
            print(f"       ↳ PATH={hier} [{c.get('chunk_type','?')}] "
                  f"score={c.get('final_score', '?'):.3f}")

        context = chunks_to_context(chunks)

        # Stage 5 — generate
        print(f"    → Calling Groq (llama-3.3-70b-versatile) …")
        try:
            text = call_groq(
                api_key             = api_key,
                context             = context,
                section_instruction = sec["instruction"],
                word_limit          = sec["word_limit"],
            )
        except Exception as e:
            print(f"    ⚠  Groq error: {e}")
            text = f"Generation failed: {e}"

        sections_content[key] = text
        print(f"    → {len(text.split())} words generated")

    # ── Save raw JSON ─────────────────────────────────────────────────────
    debug_path = os.path.join(out_dir, "generated_sections_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(sections_content, f, ensure_ascii=False, indent=2)
    print(f"✅  Debug JSON saved → {debug_path}")

    # ── Save retrieved chunks (with new metadata) ─────────────────────────
    chunks_debug_path = os.path.join(out_dir, "retrieved_chunks_debug.json")
    with open(chunks_debug_path, "w", encoding="utf-8") as f:
        json.dump(all_retrieved, f, ensure_ascii=False, indent=2)
    print(f"✅  Retrieved chunks saved → {chunks_debug_path}")

    # ── Accuracy metrics ──────────────────────────────────────────────────

    # 1. Jaccard coverage
    jaccard_scores = compute_retrieval_coverage(sections_content, all_retrieved)
    print("\n📊  Jaccard token-overlap scores (0→1):")
    for k, v in jaccard_scores.items():
        if v["jaccard"] is None:
            print(f"    {k:<15}: no chunks retrieved")
        else:
            print(f"    {k:<15}: {v['jaccard']:.4f}  "
                  f"(chunks={v['n_chunks']}, tables={v['n_table_chunks']}, "
                  f"best=§{v['best_chunk']})")

    # 2. BERT Score
    print("\n📊  BERTScore (F1 / Precision / Recall vs retrieved context):")
    bert_scores = compute_bert_scores(sections_content, all_retrieved)
    for k, v in bert_scores.items():
        if v.get("bert_f1") is None:
            reason = v.get("error", "skipped / not available")
            print(f"    {k:<15}: N/A  ({reason})")
        else:
            print(f"    {k:<15}: F1={v['bert_f1']:.4f}  "
                  f"P={v['bert_p']:.4f}  R={v['bert_r']:.4f}")

    # 3. Combined metrics JSON
    combined_metrics = {}
    for k in jaccard_scores:
        combined_metrics[k] = {
            **jaccard_scores[k],
            **bert_scores.get(k, {}),
        }

    # Summary stats
    valid_j   = [v["jaccard"]   for v in jaccard_scores.values() if v["jaccard"]  is not None]
    valid_b   = [v["bert_f1"]   for v in bert_scores.values()    if v.get("bert_f1") is not None]
    avg_j     = round(sum(valid_j) / len(valid_j), 4) if valid_j else None
    avg_b     = round(sum(valid_b) / len(valid_b), 4) if valid_b else None

    combined_metrics["_summary"] = {
        "avg_jaccard":    avg_j,
        "avg_bert_f1":   avg_b,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model":     "llama-3.3-70b-versatile (Groq)",
        "n_sections":    len(MS_SECTIONS),
    }

    print(f"\n📈  Summary — avg Jaccard: {avg_j}  |  avg BERT F1: {avg_b}")

    metric_path = os.path.join(out_dir, "accuracy_metrics.json")
    with open(metric_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"✅  Metrics saved → {metric_path}")


# ══════════════════════════════════════════════════════════════════════════
# 8.  CLI ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4+5: Retrieve from ChromaDB and generate Method Statement via Groq."
    )
    parser.add_argument("--db_dir",    default="chroma_db",
                        help="Path to persisted ChromaDB directory")
    parser.add_argument("--out_dir",   default="output",
                        help="Directory for output files")
    parser.add_argument("--api_key",   default=os.getenv("GROQ_API_KEY", ""),
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--team_name", default="Team Name / Unstop ID",
                        help="Team name for the title page")
    parser.add_argument("--n_results", type=int, default=8,
                        help="Number of chunks retrieved per query (default 8)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "Groq API key required.  Pass --api_key or set GROQ_API_KEY env var.\n"
            "Get a free key at: https://console.groq.com"
        )

    run_pipeline(
        db_dir      = args.db_dir,
        api_key     = args.api_key,
        out_dir     = args.out_dir,
        team_name   = args.team_name,
        n_per_query = args.n_results,
    )