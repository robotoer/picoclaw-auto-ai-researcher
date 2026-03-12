# 10 — Hardware Budget, API Costs, and Scaling Strategy

> *How much does it cost to run an autonomous AI researcher at each capability level, and how should compute be allocated?*

---

## 1. The Compute-Performance Relationship

Following the bitter lesson (doc 09), the system's research capability scales with compute investment. This document provides concrete budgets, specifications, and scaling strategies.

**The fundamental equation:**

```
Research_Output_Quality ≈ k × log(Compute_Budget) + baseline
```

Where `k` depends on algorithmic efficiency (how well the system converts compute into knowledge) and `baseline` depends on the quality of the base models and infrastructure.

---

## 2. Component Cost Breakdown

### 2.1 LLM API Costs (as of early 2026)

| Provider | Model | Input (per 1M tokens) | Output (per 1M tokens) | Context Window |
|---|---|---|---|---|
| Anthropic | Claude Opus 4.6 | $5.00 | $25.00 | 200K |
| Anthropic | Claude Sonnet 4.6 | $3.00 | $15.00 | 200K |
| Anthropic | Claude Haiku 4.5 | $1.00 | $5.00 | 200K |
| OpenAI | GPT-4o | $2.50 | $10.00 | 128K |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 | 128K |
| Google | Gemini 2.5 Pro | $1.25 | $10.00 | 2M |
| Google | Gemini 2.5 Flash | $0.30 | $2.50 | 1M |
| Google | Gemini 2.5 Flash-Lite | $0.10 | $0.40 | 1M |
| DeepSeek | V3.2 | $0.28 | $0.42 | 128K |

**Cost optimization levers:** Batch processing gives 50% off (Claude, OpenAI). Cache hits give up to 90% off (Claude, DeepSeek). DeepSeek V3.2 is a game-changer for budget tiers — 10–60× cheaper than frontier models with strong research reasoning.

**Token consumption estimates per research task:**

| Task | Input Tokens | Output Tokens | Typical Model | Cost/Task |
|---|---|---|---|---|
| Paper analysis (full text) | ~30K | ~5K | Sonnet | ~$0.17 |
| Hypothesis generation | ~10K | ~3K | Opus | ~$0.37 |
| Critique/debate round | ~15K | ~5K | Sonnet | ~$0.12 |
| Experiment design | ~20K | ~8K | Opus | ~$0.90 |
| Literature search + synthesis | ~50K | ~10K | Sonnet | ~$0.30 |
| Peer review simulation (3 reviewers) | ~40K | ~15K | Sonnet | ~$0.35 |
| Full research thread (end-to-end) | ~200K+ | ~60K+ | Mixed | ~$3–10 |

### 2.2 GPU Costs (Cloud)

| GPU | VRAM | Budget Providers/hr | Major Cloud/hr | Use Case |
|---|---|---|---|---|
| NVIDIA A100 (80GB) | 80GB | $1.49 (RunPod, Jarvislabs) | $3.40–3.43 (AWS/GCP) | Medium model inference, training |
| NVIDIA L40S (48GB) | 48GB | $0.40–0.86 | $1.55–1.82 (Nebius) | Inference, embeddings |
| NVIDIA H100 (80GB) | 80GB | $1.38–2.10 (Thunder, Vast, GMI) | $3.00–3.90 (GCP, AWS) | Fine-tuning, large model inference |
| NVIDIA H200 (141GB) | 141GB | $2.50–3.80 (GMI, Jarvislabs) | $10.60 (AWS/Azure) | Large model inference, RL training |
| NVIDIA B200 | 192GB | $3.99–5.98 (DataCrunch, Vast) | Not widely available yet | Next-gen training |

Spot/preemptible instances run 60–90% cheaper. 1–3 year commitments save 45–50%.

**Cloud providers (typical H100 pricing):**

| Provider | H100 On-Demand/hr | Notes |
|---|---|---|
| Thunder Compute | $1.38 | Budget leader |
| Vast.ai | $1.87 | Marketplace model |
| GMI | $2.10 | Competitive |
| RunPod | $2.00 | Budget, spot available |
| Lambda Labs | $2.50 | ML-focused, simpler |
| GCP (a3 instances) | $3.00 | Good TPU alternatives |
| AWS (p5 instances) | $3.90 | Most mature ecosystem |
| CoreWeave | $6.16 | GPU-cloud native, enterprise |

### 2.3 On-Premise GPU Costs

| GPU | Purchase Price (New) | Used/Secondary | Amortized/month (3yr) | Power + Infra/month |
|---|---|---|---|---|
| RTX 4090 (24GB) | $1,600–2,000 | N/A | ~$55 | ~$25 |
| RTX 5090 (32GB) | $2,000–2,500 | N/A | ~$70 | ~$30 |
| A100 (80GB) | $8,000–15,000 | $4,000–9,000 | ~$300 (used) | ~$35 |
| H100 SXM (80GB) | $25,000–40,000 | Declining | ~$900 | ~$50 |
| H200 (8-GPU system) | ~$315,000 | Not yet avail. | ~$8,750 | ~$500 |
| DGX H200/B200 | $400,000–500,000 | N/A | ~$14,000 | ~$1,000 |

Add 40–50% for networking, cooling, and power infrastructure on server-class GPUs. On-premise breaks even vs. cloud at roughly 500+ GPU-hours/month sustained usage. Used A100s at $4K–9K are the current sweet spot for labs bootstrapping on-premise.

### 2.4 Infrastructure Costs

| Service | Free Tier | Small | Medium | Enterprise |
|---|---|---|---|---|
| **Neo4j** (Knowledge Graph) | Community (local) | AuraDB Free | AuraDB Pro ($65+/mo) | Enterprise ($1,000+/mo) |
| **Qdrant** (Vector DB) | Local Docker | Cloud Free (1GB) | Cloud ($25–100/mo) | Dedicated ($500+/mo) |
| **Pinecone** (Vector DB) | Starter (free) | Standard ($70/mo) | Enterprise ($500+/mo) | — |
| **Object Storage** (S3/GCS) | — | $5/TB/mo | $5/TB/mo | $5/TB/mo |
| **ArXiv API** | Free | Free | Free | Free |
| **Semantic Scholar API** | Free (100 req/s) | Free | Free | Free |
| **Papers With Code API** | Free | Free | Free | Free |

### 2.5 Embedding Model Costs

| Model | Cost per 1M tokens | Notes |
|---|---|---|
| OpenAI text-embedding-3-small | $0.02 ($0.01 batch) | Best value API embedding |
| OpenAI text-embedding-3-large | $0.13 ($0.065 batch) | Higher quality |
| Mistral Embed | $0.01 | Cheapest API option |
| Self-hosted (NV-Embed-v2 on A100) | ~$0.001 | 10–20× cheaper than API |
| SPECTER / sentence-transformers | Free (self-hosted) | Open-source, research-grade |

Embedding the full ArXiv AI corpus (~1M papers × 5K tokens avg = 5B tokens):
- API: ~$100 one-time (OpenAI small)
- Self-hosted on A100: ~$5–10 one-time
- Daily incremental updates (100–1000 papers): negligible

### 2.6 Fine-Tuning Cost Estimates

| Task | Model Size | Method | Cost per Run | Time |
|---|---|---|---|---|
| Reward model | 7B–8B | QLoRA, 1× A100 | $8–20 | 1–4 hrs |
| Reward model | 13B | QLoRA, 1× A100 | $20–60 | 2–8 hrs |
| Curriculum planner | 7B | LoRA, 1× A100 | $10–30 | 1–3 hrs |
| Full fine-tune | 70B | Full, 2–4× A100 | $100–500+ | 8–24 hrs |
| Aggressive iteration (50 runs/mo) | 7B | QLoRA | $400–1,000/mo | — |

QLoRA/LoRA is 10–20× cheaper than full fine-tuning and rarely worse in practice.

---

## 3. Detailed Tier Specifications

### Tier 0: Proof of Concept ($100–300/month)

**Architecture:** Single agent, API-only, local storage.

```
┌─────────────────────────────────────────┐
│  Single Agent (Claude Sonnet API)       │
│  └── ReAct loop (observe, reason, act)  │
│                                         │
│  Local Infrastructure:                  │
│  ├── SQLite (claim storage)             │
│  ├── NetworkX (knowledge graph)         │
│  ├── FAISS (embeddings, in-memory)      │
│  └── Local filesystem (paper cache)     │
└─────────────────────────────────────────┘
```

| Item | Monthly Cost |
|---|---|
| Claude Sonnet API (~2M tokens/day input, ~500K output) | $120–200 |
| Local compute (existing laptop/desktop) | $0 |
| Storage (local) | $0 |
| ArXiv + Semantic Scholar APIs | $0 |
| **Total** | **$120–200** |

**Capabilities:**
- Process ~50 papers/day (abstract + selective full text)
- Generate ~5 hypotheses/day
- Maintain knowledge graph with ~10K entities
- Single research thread
- No RL, no fine-tuning, no experiments
- Loops available: Loop 1 (ReAct only, no RL training)

**What you learn:** Whether the basic research loop produces useful hypotheses at all. Validates the pipeline before investing more.

---

### Tier 1: Indie Researcher ($500–2,000/month)

**Architecture:** Multi-agent (3 agents), hybrid local+API, debate loop.

```
┌─────────────────────────────────────────┐
│  Agent Team:                            │
│  ├── Generator (Opus API)               │
│  ├── Critic (Sonnet API)                │
│  └── Synthesizer (Sonnet API)           │
│                                         │
│  Local GPU (RTX 4090/5090):             │
│  ├── Local LLM (Llama 3.1 70B, 4-bit)  │
│  ├── Embedding model (SPECTER)          │
│  └── Small RL policy training           │
│                                         │
│  Infrastructure:                        │
│  ├── Neo4j Community (Docker)           │
│  ├── Qdrant (Docker)                    │
│  └── 500GB local storage                │
└─────────────────────────────────────────┘
```

| Item | Monthly Cost |
|---|---|
| Claude API (Opus + Sonnet, ~5M tokens/day) | $300–600 |
| Local GPU (amortized + power) | $80–100 |
| Cloud GPU burst (fine-tuning, ~10 hrs H100/mo) | $150–400 |
| Infrastructure (local Docker) | $0 |
| Storage | ~$10 |
| **Total** | **$540–1,110** |

**Capabilities:**
- Process ~200 papers/day
- 3-agent debate loop (Generator-Critic-Synthesizer)
- ~20 hypotheses/day with critique
- Knowledge graph with ~50K entities
- Basic curriculum learning (retrained monthly)
- Small-scale experiment execution (local GPU)
- 2–3 concurrent research threads
- Loops available: Loops 1+2 (ReAct + Debate)

---

### Tier 2: Research Lab ($5,000–20,000/month)

**Architecture:** Full multi-agent team (6+ agents), population-based evolution, continuous learning.

```
┌─────────────────────────────────────────┐
│  Agent Population (5-10 variants):      │
│  ├── Literature Analyst                 │
│  ├── Hypothesis Generator               │
│  ├── Experiment Designer                │
│  ├── Critic                             │
│  ├── Synthesizer                        │
│  ├── Science Communicator               │
│  └── Statistician                       │
│                                         │
│  Compute:                               │
│  ├── API: Claude Opus (20M tokens/day)  │
│  ├── Cloud: 2-4× H100 (inference)      │
│  ├── Cloud: 4× H100 (training cluster) │
│  └── Local: Embedding + retrieval       │
│                                         │
│  Infrastructure:                        │
│  ├── Neo4j AuraDB Pro                   │
│  ├── Qdrant Cloud (dedicated)           │
│  ├── Kubernetes cluster                 │
│  └── 10TB object storage                │
└─────────────────────────────────────────┘
```

| Item | Monthly Cost |
|---|---|
| Claude API (Opus + Sonnet, ~20M tokens/day) | $1,500–3,000 |
| Cloud GPU inference (2–4× H100) | $3,000–7,000 |
| Cloud GPU training (4× H100, ~50 hrs) | $1,500–3,000 |
| Neo4j AuraDB Pro | $300–1,000 |
| Qdrant Cloud | $200–500 |
| Kubernetes cluster | $1,000–3,000 |
| Object storage (10TB) | $50 |
| Monitoring | $100 |
| **Total** | **$7,650–17,650** |

**Capabilities:**
- Process ~1,000 papers/day (full text)
- 6-agent specialist team with debate
- Population of 5–10 strategy variants (weekly evolution)
- Meta-RL curriculum (daily updates)
- Medium-scale experiments (1B+ parameter models)
- 10+ concurrent research threads
- Monthly distillation cycles
- AKDC with 2–3 cascade levels
- Loops available: All 4 loops + Novel Path 2

---

### Tier 3: Industrial Scale ($50,000–200,000+/month)

**Architecture:** Full system with thermodynamic resource allocation, 50+ agent population, continuous self-improvement.

| Item | Monthly Cost |
|---|---|
| LLM API (enterprise tier, 100M+ tokens/day) | $10,000–30,000 |
| GPU cluster (16–64× H100/H200) | $20,000–80,000 |
| Training cluster (8–32× H100) | $5,000–20,000 |
| Enterprise databases (Neo4j + Qdrant) | $3,000–15,000 |
| Kubernetes + infrastructure | $5,000–30,000 |
| Storage + networking | $1,000–3,000 |
| Engineering support | $5,000–15,000 |
| **Total** | **$49,000–193,000** |

**Capabilities:**
- Full ArXiv AI feed in real-time
- 50+ agent population with QD evolution
- Thermodynamic resource allocation
- AKDC with 5+ cascade levels
- Large-scale experiments (7B+ models)
- 5–10 research-quality outputs per week
- All loops + both novel paths operational
- Real paradigm-shift detection

---

## 4. Compute Allocation Strategy

### 4.1 Budget Distribution by Component

The optimal distribution shifts as the system matures:

**Phase 0 (Bootstrap):**

| Component | % of Budget |
|---|---|
| LLM inference (paper analysis, hypothesis generation) | 70% |
| Embedding/retrieval | 10% |
| Infrastructure | 15% |
| Training | 5% |

**Phase 2 (RL Active):**

| Component | % of Budget |
|---|---|
| LLM inference | 40% |
| RL training + fine-tuning | 30% |
| Experiment execution | 15% |
| Infrastructure | 10% |
| Embedding/retrieval | 5% |

**Phase 4 (Full System):**

| Component | % of Budget |
|---|---|
| Experiment execution | 30% |
| RL training + fine-tuning | 25% |
| LLM inference | 25% |
| Infrastructure | 10% |
| Embedding/retrieval | 5% |
| Monitoring + safety | 5% |

**Bitter lesson prediction:** As the system matures, the fraction spent on training and experimentation should increase while the fraction on inference decreases. The system's *learned* components should become more valuable than its *prompted* components.

### 4.2 Cost Optimization Strategies

1. **Tiered model routing:** Use Haiku for paper filtering, Sonnet for analysis, Opus only for novel hypothesis generation and complex reasoning. Expected savings: 40–60% vs. Opus-only.

2. **Caching and deduplication:** Cache paper analyses, embedding computations, and common reasoning chains. Papers rarely change; most queries are repeats with slight variations. Expected savings: 20–30%.

3. **Spot instances for training:** RL training and fine-tuning are checkpoint-resumable. Use spot/preemptible instances at 50–70% discount.

4. **Batched processing:** Accumulate papers and process in batches rather than one-at-a-time. Reduces API overhead and enables batch pricing. Expected savings: 10–20%.

5. **Progressive depth:** Process all papers at abstract level ($0.01/paper). Only process the top 30% at full text level ($0.17/paper). Only run full debate on the top 10% of hypotheses.

6. **Open-source model substitution:** As open-source models improve, substitute them for API models on routine tasks. The gap is closing; by 2027, most routine research tasks should be handleable by local open-source models.

### 4.3 Break-Even Analysis

When does it make sense to buy GPUs vs. rent cloud?

| Scenario | Buy | Rent |
|---|---|---|
| <20 GPU-hours/month | Rent | Rent |
| 20–200 GPU-hours/month | Either | Either |
| >200 GPU-hours/month | Buy (2–3yr payback) | Rent if burst |
| Need latest GPUs (H200, B200) | Rent | Rent |
| Predictable workload | Buy | — |
| Burst workload (training) | Buy baseline + rent burst | Rent all |

**Recommendation for each tier:**
- Tier 0: Rent (API only)
- Tier 1: Buy 1 consumer GPU + rent burst
- Tier 2: Rent cloud cluster (flexibility > cost)
- Tier 3: Buy baseline cluster + rent burst (hybrid)

---

## 5. Scaling Laws for Research Output

### 5.1 Expected Output by Tier

| Tier | Papers Processed/Day | Hypotheses/Day | Research Outputs/Month | Estimated Quality |
|---|---|---|---|---|
| 0 | 50 | 5 | 2–3 | Low (unvalidated hypotheses) |
| 1 | 200 | 20 | 5–8 | Medium (debated, critiqued) |
| 2 | 1,000 | 50 | 15–25 | High (experimentally tested) |
| 3 | 5,000+ | 200+ | 40–60 | Very high (peer-review quality) |

### 5.2 The Self-Improvement Multiplier

Once RL training is active (Tier 1+), the system's efficiency should improve over time:

```
Effective_Output(t) = Base_Output × (1 + improvement_rate)^t
```

Conservative estimate: `improvement_rate = 5%/month` (compounding). After 12 months, effective output is ~1.8× the initial rate. After 24 months, ~3.2×.

This compounds with compute scaling: doubling compute while improving efficiency by 2× gives 4× the effective output.

### 5.3 Diminishing Returns and Efficiency Frontiers

At each tier, there's a point where adding more compute provides diminishing returns because:
- **Data bottleneck:** ArXiv publishes a finite number of papers per day
- **Reward model saturation:** The reward signal becomes less informative
- **Experiment bottleneck:** Some experiments require wall-clock time (training runs) that can't be parallelized
- **Review bottleneck:** Quality review (human or simulated) is rate-limited

The transition to the next tier should happen when hitting these plateaus, not simply when budget is available.

---

## 6. API Token Budget Calculator

### Daily Token Budget by Tier

```
Tier 0: ~2.5M tokens/day
  ├── Paper analysis: 50 papers × 35K tokens = 1.75M
  ├── Hypothesis gen: 5 × 13K tokens = 65K
  ├── Overhead (routing, logging): ~600K
  └── Total: ~2.5M input + ~500K output

Tier 1: ~7M tokens/day
  ├── Paper analysis: 200 papers × 20K tokens = 4M (Sonnet)
  ├── Hypothesis gen: 20 × 13K tokens = 260K (Opus)
  ├── Debate rounds: 20 × 20K tokens = 400K (Sonnet)
  ├── Local LLM offload: ~2M (free, local GPU)
  └── Total: ~5M API input + ~2M local

Tier 2: ~25M tokens/day
  ├── Paper analysis: 1000 × 20K = 20M (mixed local + API)
  ├── Hypothesis gen: 50 × 13K = 650K (Opus)
  ├── Debate: 50 × 20K = 1M (Sonnet)
  ├── Experiment design: 10 × 28K = 280K (Opus)
  ├── Peer review: 10 × 55K = 550K (Sonnet)
  ├── Synthesis: 5 × 60K = 300K (Opus)
  └── Total: ~10M API + ~15M local
```

---

## 7. Infrastructure Architecture by Tier

### Tier 0 (Local Everything)
```
[Laptop/Desktop]
├── Python process (single agent)
├── SQLite (claims, metadata)
├── FAISS (in-memory embeddings)
├── NetworkX (knowledge graph)
└── Local filesystem (paper cache)
```

### Tier 1 (Hybrid Local + Cloud)
```
[Workstation with GPU]
├── Docker Compose
│   ├── Agent orchestrator
│   ├── Neo4j Community
│   ├── Qdrant
│   └── Local LLM (vLLM/Ollama)
├── External APIs
│   ├── Claude API (frontier reasoning)
│   ├── ArXiv API
│   └── Semantic Scholar API
└── Spot cloud (burst training)
```

### Tier 2 (Cloud Cluster)
```
[Kubernetes Cluster]
├── Agent pods (auto-scaling)
│   ├── Literature Analyst (×2)
│   ├── Hypothesis Generator (×1)
│   ├── Critic (×2)
│   ├── Experiment Designer (×1)
│   ├── Synthesizer (×1)
│   └── Orchestrator (×1)
├── Infrastructure pods
│   ├── Neo4j (3-node cluster)
│   ├── Qdrant (replicated)
│   ├── Redis (caching)
│   └── vLLM serving (2-4× H100)
├── Training jobs (scheduled)
│   ├── RL policy training
│   ├── Reward model updates
│   └── Distillation jobs
└── Monitoring
    ├── Prometheus + Grafana
    ├── SUNFIRE dashboard
    └── Cost tracking
```

---

## 8. Migration Path Between Tiers

Moving between tiers should be incremental, not a full rebuild:

| Transition | Key Changes | Estimated Migration Time |
|---|---|---|
| Tier 0 → 1 | Add Neo4j + Qdrant (Docker); add local GPU; add Critic agent | 1–2 weeks |
| Tier 1 → 2 | Move to Kubernetes; add cloud GPU; add remaining agents; start RL training | 4–6 weeks |
| Tier 2 → 3 | Scale cluster; add thermodynamic layer; population expansion | 2–3 months |

**Key principle:** Design for Tier 2 from the start, even if running at Tier 0. Use dependency injection and configuration to swap backends (SQLite → Neo4j, FAISS → Qdrant, local → cloud). The codebase should be tier-agnostic.

---

*← Back to [Bitter Lesson, Loops & Novel Paths](09-bitter-lesson-loops-and-novel-paths.md)*
