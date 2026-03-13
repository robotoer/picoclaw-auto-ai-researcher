# E01: LLM Claim Extraction Accuracy from Scientific Papers

## Overview

This experiment evaluates how accurately large language models extract factual claims from AI/ML research papers. We compare three models (Claude Sonnet, GPT-4o, and a smaller model) against a dual-annotator ground truth derived from 20 ArXiv papers, measuring precision, recall, F1, and hallucination rate.

---

## Tasks

### Task 1: Paper Selection & Download

**Dependencies:** None

Fetch 20 papers from ArXiv across four categories (5 each), drawn from cs.AI and cs.LG, published 2023--2024, English language, 30 pages or fewer.

| Category    | Count | Query Terms                        |
|-------------|-------|------------------------------------|
| Empirical   | 5     | `"benchmark evaluation"`           |
| Theoretical | 5     | `"theoretical analysis"`           |
| Survey      | 5     | `"survey"`                         |
| Methods     | 5     | `"novel method architecture"`      |

Papers are downloaded via the ArXiv API, full text extracted, and stored as structured JSON.

### Task 2: Ground Truth Annotation

**Dependencies:** Task 1

Create a gold-standard set of claims for each paper using a rigorous dual-annotator LLM protocol:

1. **Annotator A (Conservative):** Claude Opus prompted to extract only claims that are explicitly and unambiguously stated, erring on the side of exclusion.
2. **Annotator B (Comprehensive):** Claude Opus prompted to extract all plausible claims, including those requiring minor inference from the text.
3. **Inter-Annotator Agreement:** Compute Cohen's kappa between the two extraction sets (after embedding-based alignment at cosine >= 0.90).
4. **Reconciliation:** A third Claude Opus pass reviews disagreements against the annotation guidelines (see `data/annotation_guidelines.md`) and produces the final gold standard.

Each claim in the gold standard includes:
- `text` -- the claim as a single falsifiable proposition
- `type` -- one of: quantitative, methodological, comparative, existence, causal
- `source_section` -- the section of the paper where the claim originates
- `annotator_id` -- which annotator(s) flagged it and whether reconciliation was needed

### Task 3: LLM Extraction

**Dependencies:** Task 1

Run three models on all 20 papers:

| Model          | Provider   | Purpose               |
|----------------|------------|-----------------------|
| Claude Sonnet  | Anthropic  | Primary evaluation    |
| GPT-4o         | OpenAI     | Cross-provider comparison |
| Smaller model  | TBD        | Cost/accuracy tradeoff |

Parameters:
- Temperature: 0 (deterministic)
- Runs per paper per model: 3 (to measure extraction stability)
- Prompt: standardized claim-extraction prompt referencing the annotation guidelines
- Output format: JSON array matching the gold-standard schema

### Task 4: Claim Matching

**Dependencies:** Tasks 2, 3

Match extracted claims to ground truth:

1. Embed all claims (ground truth + extracted) using `sentence-transformers` (model: `all-MiniLM-L6-v2`, ~90MB).
2. Compute pairwise cosine similarity between each extracted claim and each ground-truth claim for the same paper.
3. A match is declared at cosine similarity >= 0.85.
4. Each ground-truth claim matches at most one extracted claim (Hungarian algorithm for optimal assignment when multiple candidates exceed threshold).

### Task 5: Statistical Analysis

**Dependencies:** Task 4

Compute per-model and per-category metrics:

- **Precision:** fraction of extracted claims that match a ground-truth claim
- **Recall:** fraction of ground-truth claims that are matched by an extracted claim
- **F1:** harmonic mean of precision and recall
- **Bootstrap 95% confidence intervals:** 10,000 resamples at the paper level
- **McNemar's test:** pairwise comparison between models on per-claim correct/incorrect extraction
- **Hallucination rate:** fraction of extracted claims with no ground-truth match (1 - precision), broken down by claim type
- **Extraction stability:** agreement across 3 runs per model (Fleiss' kappa)

### Task 6: Hallucination Audit

**Dependencies:** Task 5

Manual spot-check of false positives (extracted claims with no ground-truth match):

1. Sample 50 false positives per model (or all, if fewer than 50).
2. Categorize each into: (a) genuine hallucination -- fabricated claim not in paper, (b) paraphrase miss -- valid claim that the embedding match missed, (c) over-decomposition -- valid but overly granular split of a ground-truth claim, (d) scope error -- claim from excluded category (background, citation, etc.).
3. Compute adjusted hallucination rate after reclassification.

### Task 7: Visualization

**Dependencies:** Tasks 5, 6

Generate publication-quality figures:

- Grouped bar chart: precision/recall/F1 by model, with CI error bars
- Heatmap: performance by model x paper category
- Box plot: per-paper F1 distribution by model
- Confusion-style matrix: hallucination categories by model
- Scatter plot: paper length vs. extraction recall

All figures saved as PDF and PNG at 300 DPI.

### Task 8: Paper Write-up

**Dependencies:** Tasks 5, 6, 7

Full LaTeX paper (target venue: workshop or short paper):

- Abstract, Introduction, Related Work, Method, Results, Discussion, Conclusion
- All tables and figures embedded
- Compiled to PDF via `latexmk`

---

## Hardware Requirements

| Resource   | Requirement                                                                 |
|------------|-----------------------------------------------------------------------------|
| CPU        | Any modern multi-core processor (no GPU needed)                             |
| RAM        | 8 GB minimum (sentence-transformers model is ~90 MB in memory)              |
| Disk       | ~500 MB (papers, extracted text, embeddings, intermediate results)          |
| Python     | 3.11+                                                                       |
| OS         | Linux, macOS, or Windows (tested on macOS and Linux)                        |

### API Keys

| Key                 | Required | Purpose                          |
|---------------------|----------|----------------------------------|
| `ANTHROPIC_API_KEY` | Yes      | Claude Sonnet + Opus (annotation and extraction) |
| `OPENAI_API_KEY`    | No       | GPT-4o extraction (Task 3 only)  |

### Estimated API Cost

$50--$100 total, dominated by:
- Ground truth annotation (Task 2): ~$20--40 (three Opus passes over 20 papers)
- LLM extraction (Task 3): ~$20--40 (3 models x 20 papers x 3 runs)
- Hallucination audit review (Task 6): ~$5--10

### Python Dependencies

```
sentence-transformers>=2.2.0
scipy>=1.11.0
matplotlib>=3.8.0
numpy>=1.25.0
arxiv>=2.1.0
httpx>=0.25.0
anthropic>=0.40.0
openai>=1.0.0        # optional, for GPT-4o
scikit-learn>=1.3.0
pandas>=2.1.0
```

---

## Dataset Acquisition

### Source

ArXiv API, accessed via the `arxiv` Python package (`pip install arxiv`).

### Selection Criteria

- **Categories:** `cs.AI` or `cs.LG`
- **Date range:** Published 2023-01-01 to 2024-12-31
- **Language:** English
- **Length:** 30 pages or fewer
- **Exclusions:** Papers with primarily non-textual content (e.g., dataset cards with minimal prose), withdrawn papers, duplicate submissions

### Stratification

Each category uses a targeted query to ArXiv search:

| Category    | ArXiv Query                                                         |
|-------------|---------------------------------------------------------------------|
| Empirical   | `cat:cs.AI OR cat:cs.LG AND "benchmark evaluation"`                |
| Theoretical | `cat:cs.AI OR cat:cs.LG AND "theoretical analysis"`                |
| Survey      | `cat:cs.AI OR cat:cs.LG AND "survey"`                              |
| Methods     | `cat:cs.AI OR cat:cs.LG AND "novel method architecture"`           |

For each query, retrieve the top 20 results sorted by relevance, then manually (or programmatically) filter to the first 5 that meet all selection criteria.

### Storage Format

Each paper is stored as a JSON file in `data/papers/`:

```json
{
  "arxiv_id": "2312.12345",
  "title": "...",
  "authors": ["..."],
  "abstract": "...",
  "categories": ["cs.AI", "cs.LG"],
  "published": "2023-12-15",
  "page_count": 12,
  "category_label": "empirical",
  "full_text": "...",
  "pdf_url": "https://arxiv.org/pdf/2312.12345",
  "source": "arxiv_api"
}
```

### Ground Truth Format

Each paper's ground truth is stored as a JSON file in `data/ground_truth/`:

```json
{
  "arxiv_id": "2312.12345",
  "claims": [
    {
      "text": "BERT fine-tuned on SQuAD achieves 93.2 F1 on the dev set.",
      "type": "quantitative",
      "source_section": "Results",
      "annotator_id": "reconciled",
      "annotator_a": true,
      "annotator_b": true,
      "reconciliation_needed": false
    }
  ],
  "annotation_metadata": {
    "annotator_a_count": 42,
    "annotator_b_count": 67,
    "reconciled_count": 55,
    "cohens_kappa": 0.73
  }
}
```

---

## Directory Structure

```
experiments/E01/
  README.md                  # This file
  data/
    annotation_guidelines.md # Claim annotation guidelines
    papers/                  # Downloaded paper JSON files
    ground_truth/            # Gold-standard claim annotations
    extractions/             # LLM extraction outputs
    embeddings/              # Cached sentence-transformer embeddings
  scripts/
    01_download_papers.py
    02_annotate_ground_truth.py
    03_extract_claims.py
    04_match_claims.py
    05_analyze.py
    06_audit_hallucinations.py
    07_visualize.py
  results/
    figures/
    tables/
  paper/
    main.tex
```
